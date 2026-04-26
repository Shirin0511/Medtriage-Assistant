from pydantic import BaseModel, field_validator
from enum import Enum
from typing import List, Optional
from datetime import datetime
import json
import os

class UrgencyLevel(str, Enum):

    """
    Urgency can ONLY be one of these 4 values — nothing else.
    Inheriting from str means it serializes cleanly to JSON as a string.
    
    LOW       = monitor at home, no immediate action needed
    MODERATE  = see a doctor within 24-48 hours
    HIGH      = go to urgent care today
    EMERGENCY = call emergency services immediately
    """
    
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EMERGENCY = "EMERGENCY"


class TriageReport(BaseModel):

    """
    A fully validated triage report.
    Pydantic checks every field's type and constraints automatically.
    If any field is invalid, it raises a ValidationError with clear details.
    """

    # When the report was generated — auto-set, user can't change this
    timestamp: str

    # Patient information collected during conversation
    symptoms: list[str]
    duration: str
    severity: str
    pre_existing_conditions: str

    # UrgencyLevel enum - only LOW/MODERATE/HIGH/EMERGENCY are valid
    urgency: UrgencyLevel

    # 2-3 possible conditions
    possible_conditions: List[str]

    # What the user should do next
    recommended_action: str

    # Red flag symptoms that would increase urgency
    see_doctor_if: List[str]

    # Always present — can never be removed
    disclaimer: str = (
        "This is not a medical diagnosis. "
        "Please consult a qualified healthcare professional."
    )

    # Confidence score — how well the RAG results matched the symptoms
    # Must be between 0.0 and 1.0
    confidence_score: float

    # The raw LLM response 
    raw_llm_response: str

    @field_validator("possible_conditions")
    @classmethod
    def limit_conditions(cls, v):

        """
        Validator that runs automatically when a TriageReport is created.
        Ensures we never show more than 3 possible conditions.
        Too many conditions would overwhelm and confuse the user.
        """

        if len(v)>3:
            return v[:3]
        
        return v
    
    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v):

        """
        Ensures confidence score is always between 0.0 and 1.0.
        Raises a clear error if someone passes an invalid value.
        """

        if not 0.0<= v <= 1.0: 
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {v}")
        
        return round(v,2)


def parse_urgency(llm_response:str) -> UrgencyLevel:

    """
    Scans the LLM response for the urgency level keyword.
    Returns the matching UrgencyLevel enum value.
    Defaults to MODERATE if nothing is found — better safe than sorry.
    """

    response_upper = llm_response.upper()

    # Check from most to least severe
    # Order matters — check EMERGENCY before HIGH before MODERATE before LOW

    if "EMERGENCY" in response_upper:
        return UrgencyLevel.EMERGENCY
    elif "HIGH" in response_upper:
        return UrgencyLevel.HIGH
    elif "MODERATE" in response_upper:
        return UrgencyLevel.MODERATE
    elif "LOW" in response_upper:
        return UrgencyLevel.LOW
    
    else:
        # LLM didn't include a clear urgency level
        # Default to MODERATE — we never want to under-triage
        return UrgencyLevel.MODERATE


def parse_possible_conditions(llm_response: str) -> List[str]:

    """
    Extracts the list of possible conditions from the LLM response.
    Looks for the POSSIBLE CONDITIONS section and pulls out bullet points.
    """       

    lines = llm_response.split("\n")

    conditions=[]

    in_conditions_section= False

    for line in lines:
        line = line.strip()

        # Detect when we enter the conditions section
        if "POSSIBLE CONDITIONS" in line.upper():
            in_conditions_section = True
            continue

        # Detect when we leave the conditions section
        if in_conditions_section and line.upper().startswith(("RECOMMENDED","SEE A DOCTOR", "DISCLAIMER","URGENCY")):
            break


        # Extract bullet point lines inside the section
        # Bullet points start with -, *, or a number like "1."
        if in_conditions_section and line and (
            line.startswith("-") or
            line.startswith("*") or
            (len(line)>2 and line[0].isdigit() and line[1] in '.)')
        ):
            # Remove the bullet character and clean up whitespace
            condition = line.lstrip('-*•123456789.) ').strip()
            if condition:
                conditions.append(condition)

    #Fallback
    if not conditions:
        conditions = ['Symptoms require professional medical evaluation']  

    return conditions   

def parse_see_doctor_if(llm_response: str) -> List[str]:

    """
    Extracts the red flag symptoms from the SEE A DOCTOR IF section.
    Same pattern as parse_possible_conditions but for a different section.
    """

    flags = []
    in_section = False
    lines = llm_response.split("\n")

    for line in lines:
        line = line.strip()

        if "SEE A DOCTOR" in line.upper():
            in_section = True
            continue

        if in_section and line.upper().startswith(("DISCLAIMER", "RECOMMENDED", "URGENCY", "POSSIBLE")):
            break

        if in_section and line and (
            line.startswith('-') or 
            line.startswith('*') or
            (len(line)>2 and line[0].isdigit and line[1] in '.)')
        ) :
            flag = line.lstrip('-*•123456789.) ').strip()
            if flag:
                flags.append(flag)


    if not flags:
        flags= ["Symptoms worsen significantly", "New symptoms develop"]

    return flags    


def parse_recommended_action(llm_response: str) -> str:

    """
    Extracts the recommended action text from the LLM response.
    Returns everything between RECOMMENDED ACTION and the next section.
    """
     
    lines = llm_response.split("\n")
    in_section = False
    action_lines = []

    for line in lines:
        line= line.strip()

        if "RECOMMENDED ACTION" in line.upper():
            in_section = True
            continue
         
        if in_section and line.upper().startswith(("SEE A DOCTOR", "DISCLAIMER", "POSSIBLE", "URGENCY")):
            break
         
        
        if in_section and line:
            action_lines.append(line) 

    if action_lines:
        return " ".join(action_lines)

    return "Please consult a healthcare professional for proper evaluation."


# Checking how relevant were the RAG results?

def calculate_confidence(
        rag_results : list,
        conversation_summary : dict
) -> float:
    
    """
    Calculates a confidence score between 0.0 and 1.0.
    
    This is a heuristic (rule-based estimate), not a precise ML score.
    It measures how well the retrieved medical knowledge matches
    the user's reported symptoms.
    
    Higher score = RAG found more relevant results = more confident report
    """

    if not rag_results:
        return 0.1 # no results = very low confidence
    
    score= 0.0

    # Each result contributes up to 0.25 to the score
    score += min(len(rag_results) * 0.25, 0.75)

    # Bonus: check if symptom keywords appear in the matched questions
    # This means the RAG found truly relevant content, not just similar-sounding text
    symptoms_text = " ".join(conversation_summary.get("summary",[])).lower()
    symptoms_words = set(symptoms_text.split())

    matched_keywords = 0
    for result in rag_results:
        matched_q = result.get("question","").lower()
        # Count how many symptom words appear in the matched question
        for word in symptoms_words:
            if len(word)>3 and word in matched_q:
                matched_keywords+=1

    # Each keyword match adds a small bonus, capped at 0.25
    keyword_score = min(matched_keywords*0.05, 0.25)      
    score+=keyword_score

    return min(round(score,2),1.0)


#Main Report Builder

def build_report(llm_response:str, conversation_summary:dict, rag_results:list) -> TriageReport:

    """
    Takes the raw LLM response + collected data and builds
    a fully validated TriageReport Pydantic object.
    
    If any field fails validation, Pydantic raises a clear error.
    """

     # === DIAGNOSTIC: see what the parser actually receives ===
    print("\n" + "=" * 60)
    print("LLM RESPONSE AS RECEIVED BY build_report:")
    print(repr(llm_response[:500]))   # repr() reveals hidden characters
    print("=" * 60 + "\n")
    # === END DIAGNOSTIC ===


    # Parse each section of the LLM response
    urgency = parse_urgency(llm_response)
    conditions = parse_possible_conditions(llm_response)
    see_doctor_if = parse_see_doctor_if(llm_response)
    recommended_action = parse_recommended_action(llm_response)
    confidence = calculate_confidence(rag_results, conversation_summary)

    # === DIAGNOSTIC PRINTS ===
    print("\n" + "=" * 60)
    print("PARSER OUTPUT:")
    print(f"  urgency: {urgency}")
    print(f"  conditions ({len(conditions)}): {conditions}")
    print(f"  see_doctor_if ({len(see_doctor_if)}): {see_doctor_if}")
    print(f"  recommended_action: {recommended_action[:100]}...")
    print(f"  confidence: {confidence}")
    print("=" * 60 + "\n")
    # === END DIAGNOSTIC ===

    # Build and return the validated Pydantic object
    # Pydantic validates ALL fields automatically when this is called

    report = TriageReport(
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        symptoms = conversation_summary.get("symptoms",[]),
        duration = conversation_summary.get("duration","not specified"),
        severity = conversation_summary.get("severity","not specified"),
        pre_existing_conditions = conversation_summary.get("pre_existing_conditions","none"),
        urgency=urgency,
        possible_conditions=conditions,
        recommended_action=recommended_action,
        see_doctor_if=see_doctor_if,
        confidence_score=confidence,
        raw_llm_response=llm_response
    )

    return report

def export_report(report: TriageReport, output_dir: str= "data/processed")-> str:
    """
    Exports the triage report as a JSON file.
    This is the "downloadable report" feature of the app.
    
    Returns the path to the saved file.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Use timestamp in filename so reports don't overwrite each other

    filename = f"triage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)

    # model_dump() converts the Pydantic object to a plain Python dict
    # Then json.dump() writes it to a file
    with open(filepath, "w") as f:
        json.dump(report.model_dump(), f, indent=2)

    print(f"Report saved to {filepath}")
    return filepath


if __name__ == "__main__":

    print("=== Testing Report Builder ===\n")

    # Simulate a realistic LLM response
    sample_llm_response = """
URGENCY LEVEL: MODERATE

POSSIBLE CONDITIONS:
- Symptoms may suggest a viral infection such as influenza
- Could indicate a bacterial infection requiring medical attention
- May suggest sinusitis given the combination of headache and fever

RECOMMENDED ACTION:
Visit a GP or urgent care clinic within 24 hours. Rest and stay hydrated
in the meantime. Monitor your temperature regularly.

SEE A DOCTOR IMMEDIATELY IF:
- Fever rises above 103 degrees F or 39.4 degrees C
- Headache becomes sudden and extremely severe
- You develop difficulty breathing given your asthma history

DISCLAIMER:
This is not a medical diagnosis. Please consult a qualified healthcare professional.
"""

    sample_summary = {
        "symptoms": ["bad headache and high fever"],
        "duration": "2 days",
        "severity": "7 out of 10",
        "pre_existing_conditions": "mild asthma"
    }
    sample_rag = [
        {"matched_question": "What are symptoms of fever and headache?",
         "answer": "Fever and headache together may indicate viral or bacterial infection."},
        {"matched_question": "When should I see a doctor for headache?",
         "answer": "See a doctor if headache is severe or accompanied by high fever."}
    ]

    # Build the report
    report = build_report(sample_llm_response, sample_summary, sample_rag)

    # Print the validated report
    print("=== VALIDATED TRIAGE REPORT ===\n")
    print(f"Timestamp:      {report.timestamp}")
    print(f"Urgency:        {report.urgency.value}")
    print(f"Confidence:     {report.confidence_score}")
    print(f"Symptoms:       {report.symptoms}")
    print(f"Duration:       {report.duration}")
    print(f"Severity:       {report.severity}")
    print(f"Conditions:     {report.pre_existing_conditions}")
    print(f"\nPossible Conditions:")
    for c in report.possible_conditions:
        print(f"  - {c}")
    print(f"\nRecommended Action:\n  {report.recommended_action}")
    print(f"\nSee Doctor If:")
    for f in report.see_doctor_if:
        print(f"  - {f}")
    print(f"\nDisclaimer: {report.disclaimer}")

    # Export to JSON
    print("\n=== Exporting Report ===")
    path = export_report(report)
    print(f"Exported to: {path}")








            




     
     
    
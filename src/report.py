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
    conditions: str

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
    lines = llm_response.split("/n")

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


            




     
     
    
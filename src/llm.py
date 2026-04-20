import requests
import os
import json
from dotenv import load_dotenv


load_dotenv()

# --------------------------------------------------------------------------
# PART 1: CONFIGURATION
# --------------------------------------------------------------------------


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-8b-8192"

SYSTEM_PROMPT = """You are MedTriage, a medical symptom triage assistant.

Your job is to:
1. Analyze the user's symptoms based on the information collected
2. Retrieve relevant medical knowledge provided to you
3. Output a structured triage assessment

You must ALWAYS follow these rules:
- NEVER diagnose any condition with certainty
- NEVER recommend specific medications or dosages  
- NEVER claim to replace a real doctor
- ALWAYS use hedged language like "may suggest", "could indicate", "possible"
- ALWAYS recommend seeing a doctor for any moderate or high urgency case
- If someone describes emergency symptoms (chest pain, difficulty breathing, 
  loss of consciousness), tell them to call emergency services IMMEDIATELY

You are a triage assistant, not a doctor. Your role is to help users 
understand the urgency of their symptoms and whether they need medical care.
"""

def call_groq(messages: list, system_prompt: str = SYSTEM_PROMPT) -> str:

    """
    Sends a conversation to the Groq API and returns the response text.
    
    messages    = list of {"role": "user"/"assistant", "content": "..."}
    system_prompt = instructions that define the bot's behaviour
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Please add it to your .env file"
        )
    
    # Build the headers — Authorization tells Groq who we are
    headers = {
        "authorization" : f"Bearer {GROQ_API_KEY}",
        "Content-Type" : "application/json"
    }

    # Build the request body
    body = {
        "model" : MODEL,

        "messages" : [
            {
                "role" : "system",
                "content" : system_prompt
            }
        ] + messages,

        "temperature" : 0,

        "max_tokens" : 1024
    }

    # Make the POST request to Groq
    # timeout=30 means give up if no response in 30 seconds
    response = requests.post(GROQ_URL, headers=headers, json=body, timeout=30)

    # If Groq returns an error (e.g. invalid key, rate limit)
    # raise_for_status() will throw an exception with the error details
    response.raise_for_status()

    data = response.json()

    # Extract just the text content from the response
    # data["choices"][0] = first (and only) response option
    # ["message"]["content"] = the actual text the LLM generated
    return data['choices'][0]['message']['content']


def generate_triage_response(conversation_summary : dict, conversation_history: list, rag_results : list) -> str:

    """
    This is the main function called after all 4 turns are complete.
    
    It takes:
        conversation_summary  = dict from get_conversation_summary() in Phase 4
                                contains symptoms, duration, severity, conditions
        
        rag_results           = list from search() in Phase 3
                                contains relevant medical Q&A pairs
        
        conversation_history  = full chat history from ConversationState
                                so the LLM knows everything that was said
    
    It returns:
        A structured triage response string from the LLM
    """

    # --- Step 1: Format the RAG results into readable text ---
    # The LLM needs the retrieved medical knowledge as plain text
    # We format each result clearly so the LLM can reference it
    rag_context = ""

    for i, result in enumerate(rag_results):
        rag_context += f"---Medical Reference {i+1}---\n"
        rag_context += f"Question: {result['question']}\n"
        rag_context += f"Answer: {result['answer']}\n"

    # --- Step 2: Format the collected patient info ---
    symptoms_text = " ".join(conversation_summary.get("symptoms",[]))
    duration = conversation_summary.get("duration","not specified")
    severity = conversation_summary.get("severity", "not specified")
    conditions = conversation_summary.get("pre-existing conditions","not specified")    


    # --- Step 3: Build the triage prompt ---
    # This is the instruction we send to the LLM along with all the context
    # The more specific this prompt, the more structured the output
    triage_prompt = f"""
Based on the following patient information and medical references, 
provide a structured triage assessment.

PATIENT INFORMATION:
- Symptoms: {symptoms_text}
- Duration: {duration}
- Severity (self-reported): {severity}
- Pre-existing conditions: {conditions}

RELEVANT MEDICAL REFERENCES:
{rag_context}

Please provide your triage assessment in the following format:

URGENCY LEVEL: [LOW / MODERATE / HIGH / EMERGENCY]

POSSIBLE CONDITIONS:- (list 2-3 possible conditions using hedged language like "may suggest" or "could indicate")

RECOMMENDED ACTION:
(What the person should do — e.g. monitor at home, see a GP, go to urgent care, call emergency services)

SEE A DOCTOR IMMEDIATELY IF:
- (list 2-3 red flag symptoms that would increase urgency)

DISCLAIMER:
This is not a medical diagnosis. Please consult a qualified healthcare professional.
"""
    
    # --- Step 4: Add the triage prompt to conversation history ---
    # We append it as a user message so the LLM sees it as the final instruction

    message = conversation_history + [
        {
            "role" : "user",
            "content" : triage_prompt
        }
    ]

    # Step-5 Call the LLM
    response = call_groq(message)
    return response


if __name__ == "__main__":

    print("=== Testing LLM Integration ===\n")

    # Simulate what Phase 4 would pass in
    test_summary = {
        "symptoms": ["bad headache and high fever"],
        "duration": "2 days",
        "severity": "7 out of 10",
        "pre_existing_conditions": "mild asthma",
        "ready_for_triage": True
    }

    # Simulate what Phase 3 would return
    test_rag_results = [
        {
            "matched_question": "What are the symptoms of Q Fever?",
            "answer": "Q fever can cause high fevers up to 104-105F, severe headache, general malaise, myalgia, chills and sweats."
        },
        {
            "matched_question": "What is the outlook for Headache?",
            "answer": "Not all headaches require medical attention. Sudden severe headache or headache with fever may signal serious conditions."
        }
    ]

    # Simulate a minimal conversation history
    test_history = [
        {"role": "user", "content": "I have a bad headache and high fever"},
        {"role": "assistant", "content": "How long have you had these symptoms?"},
        {"role": "user", "content": "2 days"},
        {"role": "assistant", "content": "How severe on a scale of 1-10?"},
        {"role": "user", "content": "7 out of 10"},
        {"role": "assistant", "content": "Any pre-existing conditions?"},
        {"role": "user", "content": "mild asthma"},
    ]

    # Generate the triage report
    result = generate_triage_response(test_summary, test_rag_results, test_history)

    print("=== TRIAGE REPORT ===\n")
    print(result)
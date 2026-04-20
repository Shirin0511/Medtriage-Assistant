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
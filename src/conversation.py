from dataclasses import dataclass, field
from typing import Optional, List
import re



#State Object

@dataclass
class ConversationState:

    """
    One instance of this class = one user's session.
    Every field starts empty and gets filled as the conversation progresses.
    """


    #patients symptoms
    symptoms : List[str] = field(default_factory=list)

    #how long has the patient had those symptoms
    duration : Optional[str] = None

    #how severe the symptoms are on a scale of 1-10
    severity : Optional[str] = None

    #any pre-existing conditions like diabetes, low/high BP
    conditions : Optional[str] = None

    # Full chat history — list of {"role": "user"/"assistant", "content": "..."}
    # This gets sent to the LLM so it has full context of the conversation
    history :List[str] = field(default_factory=list)

    #this will tell us what question to ask next
    turn : int = 0

    #once enough info has been collected, the pipeline should be triggered
    trigger_triage : bool = False




# Follow-up Questions
# These are asked one per turn to collect structured info from the user
# Each question targets a specific piece of missing information
# The order matters — we ask the most important things first
FOLLOWUP_QUESTIONS = [
    # Turn 0 → we just got symptoms, now ask duration
    "How long have you had these symptoms?",

    # Turn 1 → we have duration, now ask severity
    "On a scale of 1 to 10, how severe are your symptoms? (1 = mild, 10 = unbearable)",

    # Turn 2 → we have severity, now ask about pre-existing conditions
    "Do you have any pre-existing medical conditions or allergies we should know about? (Type 'none' if not)",
]


def start_conversation() -> ConversationState:

    """
    Called when a new user session begins.
    Returns a fresh empty ConversationState object.
    Think of this like pressing "New Chat".
    
    """

    return ConversationState()



def process_turn(state: ConversationState, user_msg: str):

    """
    
    The main function — called every time the user sends a message.
    
    Takes:
        state        = the current ConversationState object
        user_message = what the user just typed
    
    Returns:
        (updated_state, bot_response)
        - updated_state = ConversationState with new info added
        - bot_response  = the string the bot should say next

    """
    # --- Step 1: Add the user's message to history ---
    state.history.append(
        {
            "role" :"user",
            "content" : user_msg
        }
    )


     # --- Step 2: Extract information based on which turn we're on ---
    # Turn 0 = user just described their symptoms (first message)
    # Turn 1 = user just answered "how long?" question
    # Turn 2 = user just answered "how severe?" question
    # Turn 3 = user just answered "any conditions?" question

    if state.turn == 0:
        # Validate symptoms
        is_valid, err_msg = validate_symptoms(user_msg)

        if not is_valid:
            bot_response = err_msg
            state.history.append({"role":"assistant", "content":bot_response})
            return state, bot_response
        
        # First message from user — treat it as their symptom description
        state.symptoms.append(user_msg)
        bot_response = FOLLOWUP_QUESTIONS[0] # ask "how long?"

    elif state.turn == 1:

        # Validate duration
        is_valid, err_msg = validate_duration(user_msg)

        if not is_valid:
            bot_response = err_msg
            state.history.append({"role":"assistant", "content":bot_response})
            return state, bot_response
        
        # User answered the duration question
        state.duration = user_msg
        bot_response = FOLLOWUP_QUESTIONS[1] # ask "how severe?"

    elif state.turn ==2:

        # Validate severity
        is_valid, err_msg = validate_severity(user_msg)

        if not is_valid:
            bot_response = err_msg
            state.history.append({"role":"assistant", "content":bot_response})
            return state, bot_response
       
        # User answered the severity question
        state.severity = user_msg
        bot_response = FOLLOWUP_QUESTIONS[2]  # ask about conditions?

    elif state.turn ==3:

        # User answered the conditions question
        state.conditions = user_msg

        # We now have all 4 pieces of information:
        # Flip the flag to signal the triage pipeline to run
        state.trigger_triage = True

        # Tell the user we're generating their report
        bot_response = (
            "Thank you for sharing that information. "
            "Let me analyze your symptoms and prepare a triage report for you..."
        )

    else:

        # Fallback — shouldn't normally reach here
        # But if it does, we just prompt the report generation

        state.trigger_triage = True
        bot_response = "Analyzing your symptoms"


    # --- Step 3: Add the bot's response to history ---
    # This keeps history complete — every user message has a bot reply after it

    state.history.append(
        {
            "role" : "assistant",
            "content" : bot_response
        }
    )

    #Increment the turn counter

    state.turn +=1

    return state, bot_response


def build_rag_query( state: ConversationState) -> str:

    """
    Once we have all the info, we build a single rich query string
    to send to the RAG pipeline (vectorstore search).
    
    We combine all collected info into one sentence so the embedding
    model can find the most relevant medical Q&A pairs.
    """

    query_parts= []

    if state.symptoms:
        query_parts.append(f"Symptoms : {' '.join(state.symptoms)}")

    if state.duration:
        query_parts.append(f"Duration : {state.duration}")

    if state.severity:
        query_parts.append(f"Severity : {state.severity}")

    if state.conditions:
        query_parts.append(f"Conditions : {state.conditions}")

    # Join everything into one query string
    # e.g. "Symptoms: headache and fever Duration: 2 days Severity: 7/10 ..."
    return " | ".join(query_parts) 

def get_conversation_summary(state: ConversationState) -> dict:

    """
    Returns a clean dictionary summary of everything collected.
    Used by the report generator (Phase 6) to know what info to include.
    """

    return {
        "symptoms" : state.symptoms,
        "duration" : state.duration,
        "severity" : state.severity,
        "pre_existing_conditions" : state.conditions,
        "Turn" : state.turn,
        "Trigger Triage" : state.trigger_triage
    }


# --- Validation helpers ---

def validate_duration(text:str) -> tuple[bool,str]:

    """
    Checks if the input looks like a plausible duration.
    Returns (is_valid, error_message).
    """

    text = text.strip().lower()

    if len(text) < 2:
        return False, "Please provide a more specific duration (e.g., '2 days', '3 hours', '1 week')."

    # Must contain at least one digit OR a known time word
    has_digit = bool(re.search(r'\d',text))
    time_words = ['hour', 'day', 'week', 'month', 'year', 'minute',
                  'morning', 'evening', 'night', 'yesterday', 'today',
                  'since', 'few', 'couple', 'several']
    
    has_time_word = any(word in text for word in time_words)

    if not(has_digit or has_time_word):
        return False, (
            "I couldn't understand that duration. "
            "Please tell me how long in days, hours, or weeks "
            "(e.g., '2 days', 'about a week', 'since yesterday')."
        )
    
    return True, ""


def validate_severity(text: str) -> tuple[bool, str]:

    """
    Checks if severity is a number 1-10 or a recognized severity word.
    Returns (is_valid, error_message).
    """

    text = text.strip().lower()

    #Extracting a number
    numbers = re.findall(r'\d',text)

    if numbers:
        n= int(numbers[0])
        if 1 <= n <= 10:
            return True, ""
        
        else:
            return False, "Please provide severity between 1 and 10"
        

def validate_symptoms(text: str) -> tuple[bool, str]:

    """
    Basic check that symptoms description is meaningful.
    """

    if len(text) < 3:
        return False, "Please describe your symptoms in a bit more detail."

    # Reject if it's just random characters with no vowels (gibberish heuristic)
    if not re.search(r'[aeiouAEIOU]', text):
        return False, "I couldn't understand that. Please describe your symptoms in plain words."  

    return True, ""


if __name__ == "__main__":

    print("=== Testing Conversation Manager ===\n")

    # Start a fresh session
    state = start_conversation()

    # Simulate a full 4-turn conversation
    test_inputs = [
        "I have a bad headache and high fever",  # turn 0 — symptoms
        "About 2 days",                           # turn 1 — duration
        "7 out of 10",                            # turn 2 — severity
        "I have mild asthma"                      # turn 3 — conditions
    ]

    for user_input in test_inputs:
        print(f"User: {user_input}")
        state, response = process_turn(state, user_input)
        print(f"Bot:  {response}")
        print()

    # Show what we collected
    print("=== Collected Information ===")
    summary = get_conversation_summary(state)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Show the RAG query that would be built
    print(f"\n=== RAG Query ===")
    print(build_rag_query(state))    





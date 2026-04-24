import gradio as gr
import os
from dotenv import load_dotenv
from src.conversation import start_conversation, process_turn, build_rag_query, get_conversation_summary, ConversationState
from src.vectorstore import search, get_collection
from src.llm import generate_triage_response
from src.safety import safety_check, check_llm_response
from src.report import build_report, export_report


load_dotenv()

def chat(user_message: str, history: list, state:dict) -> tuple:

    """
    The main function Gradio calls on every user message.

    Takes:
        user_message = what the user just typed
        history      = list of [user_msg, bot_msg] pairs for display
        state        = our ConversationState stored as a dict

    Returns:
        ("", updated_history, updated_state, report_text, report_visible)
        - ""               = clears the input textbox after sending
        - updated_history  = chat history with new messages added
        - updated_state    = updated conversation state
        - report_text      = the triage report (shown after turn 4)
        - report_visible   = whether to show the report panel
    """

    #if user message is empty we should not process that
    if not user_message.strip():
        return "", history, state, "", gr.update(visible=False)
    

    # Gradio stores state as a plain dict between calls, we need to reconstruct our ConversationState object from it
    convo_state = ConversationState(**state)

    #Run safety checks
    safety_result = safety_check(user_message)

    if not safety_result['safe']:

        #input is dangerous - return response immediately, dont call llm and no state update
        history.append({"role":"user","content":user_message})
        history.append({"role":"assistant","content":safety_result['response']})

        return "", history, state, gr.update(value="",visible=False)
    
    #Process the turn through conversation manager
    convo_state, bot_response = process_turn(convo_state, user_message)

    #Add the response to history
    history.append({"role":"user","content":user_message})
    history.append({"role":"assistant","content":bot_response})

    #Check if we have enough info to generate triage report
    if convo_state.trigger_triage:
    
        # Build the RAG search query from collected symptoms
        rag_query = build_rag_query(convo_state)

        #Search ChromaDB for relevant medical knowledge
        rag_results = search(rag_query, n_results=3)

        #Get the conversation summary dict
        summary = get_conversation_summary(convo_state)

        #Generate LLM response
        llm_response = generate_triage_response(summary, convo_state.history, rag_results)

        #Check LLM output safety
        output_safety = check_llm_response(llm_response)

        if not output_safety['safe']:
            llm_response= output_safety['response']


        #Build validated pydantic report
        report = build_report(llm_response, summary, rag_results)    

        #Exporting the report
        export_report(report)

        # Format the report for display in the UI
        report_display = format_report_for_display(report)

        # Save report to state so download button can access it
        state_dict = convo_state.__dict__.copy()
        state_dict['last_report'] = report.model_dump()

        return "", history, state_dict, gr.update(value=report_display, visible=True)
    
     # Not ready for triage yet — just return updated state
    return "", history, convo_state.__dict__, gr.update(visible=False)



# Converts TriageReport object into a clean readable string for the UI
def format_report_for_display(report) -> str:

    """
    Takes a TriageReport Pydantic object and formats it as
    a clean readable string for the Gradio UI.
    """

    # Confidence bar — visual representation of confidence score
    # Each █ represents 10% confidence
    filled = int(report.confidence_score * 10)
    empty = 10 - filled
    confidence_bar = "█" * filled + "░" * empty

    # Urgency emoji mapping
    urgency_emoji = {
        "LOW": "🟢",
        "MODERATE": "🟡",
        "HIGH": "🟠",
        "EMERGENCY": "🔴"
    }
    emoji = urgency_emoji.get(report.urgency.value, "🟡")

    # Build the display string
    display = f"""
╔══════════════════════════════════════╗
         TRIAGE REPORT
╚══════════════════════════════════════╝

{emoji} URGENCY LEVEL: {report.urgency.value}

📊 CONFIDENCE: {confidence_bar} ({int(report.confidence_score * 100)}%)

👤 PATIENT SUMMARY:
   • Symptoms:   {', '.join(report.symptoms)}
   • Duration:   {report.duration}
   • Severity:   {report.severity}
   • Conditions: {report.pre_existing_conditions}

🔍 POSSIBLE CONDITIONS:
"""
    for condition in report.possible_conditions:
        display += f"   • {condition}\n"

    display += f"""
✅ RECOMMENDED ACTION:
   {report.recommended_action}

⚠️  SEE A DOCTOR IMMEDIATELY IF:
"""
    for flag in report.see_doctor_if:
        display += f"   • {flag}\n"

    display += f"""
📅 Generated: {report.timestamp}

⚕️  DISCLAIMER:
   {report.disclaimer}
"""
    return display


#Helper functions for buttons
def reset_conversation():
    """
    Called when user clicks 'New Consultation' button.
    Resets everything back to the initial state.
    """
    # Fresh conversation state
    fresh_state = start_conversation()

    # Welcome message
    welcome = [
    {"role": "assistant", "content": "👋 Hello! I'm MedTriage, your symptom triage assistant.\n\nPlease describe your symptoms and I'll help you understand how urgently you need medical care.\n\n⚠️ Note: I am NOT a doctor. I cannot diagnose conditions or prescribe medications. For emergencies, call 112 immediately."}
]
    return welcome, fresh_state.__dict__, gr.update(visible=False)


def get_initial_state():
    """Returns the initial conversation state as a dict for Gradio."""
    return start_conversation().__dict__


#THE GRADIO UI

def build_ui():
    """
    Builds and returns the Gradio interface.
    We use gr.Blocks() for full layout control instead of gr.Interface()
    which is more limited.
    """

    with gr.Blocks(
        title="MedTriage Assistant",
        # Custom CSS for a cleaner look
        css="""
        .report-box {
            font-family: monospace;
            font-size: 14px;
        }
        .disclaimer {
            color: #888;
            font-size: 12px;
        }
        """
    ) as demo:

        # --- Header ---
        gr.Markdown("""
        # 🏥 MedTriage Assistant
        ### AI-Powered Symptom Triage
        
        Describe your symptoms and answer a few questions. 
        I'll help you understand the urgency of your condition.
        
        > ⚠️ **This tool does not replace professional medical advice.
        For emergencies, call 112 (India) or 911 (US) immediately.**
        """)

        # --- State (invisible — stores conversation between turns) ---
        # gr.State() persists data between Gradio function calls
        # We initialize it with a fresh ConversationState dict
        state = gr.State(value=get_initial_state())

        # --- Main Layout: Two columns ---
        with gr.Row():

            # Left column: Chat interface
            with gr.Column(scale=3):
                gr.Markdown("### 💬 Consultation")

                # Chatbot displays the conversation history
                # type="messages" uses the newer message format
                chatbot = gr.Chatbot(
    value=[
        {"role": "assistant", "content": "👋 Hello! I'm MedTriage, your symptom triage assistant.\n\nPlease describe your symptoms and I'll help you understand how urgently you need medical care.\n\n⚠️ Note: I am NOT a doctor. I cannot diagnose conditions or prescribe medications. For emergencies, call 112 immediately."}
    ],
    height=400,
    label="Conversation",
)

                # Input row: textbox + send button side by side
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Describe your symptoms here...",
                        label="Your message",
                        scale=4,        # takes up 4x more space than button
                        lines=1
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                # New consultation button — resets everything
                reset_btn = gr.Button(
                    "🔄 New Consultation",
                    variant="secondary"
                )

            # Right column: Triage Report
            with gr.Column(scale=2):
                gr.Markdown("### 📋 Triage Report")

                # Report display — hidden until triage is ready
                report_output = gr.Textbox(
                    label="Your Triage Assessment",
                    lines=25,
                    interactive=False,  # user can't edit this
                    visible=False,      # hidden initially
                    elem_classes=["report-box"]
                )

                # Shown once report is generated
                gr.Markdown(
                    "Your triage report will appear here after "
                    "the consultation is complete.",
                    visible=True
                )

        # --- Wire up the events ---
        # Both pressing Enter in textbox AND clicking Send trigger chat()

        # When user presses Enter in the textbox
        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot, state],
            outputs=[msg_input, chatbot, state, report_output]
        )

        # When user clicks Send button
        send_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot, state],
            outputs=[msg_input, chatbot, state, report_output]
        )

        # When user clicks New Consultation
        reset_btn.click(
            fn=reset_conversation,
            inputs=[],
            outputs=[chatbot, state, report_output]
        )

    return demo


#Main Function

if __name__ == "__main__":
    print("Starting MedTriage Assistant...")
    print("Loading vectorstore...")

    get_collection()

    print("Vectorstore ready!")
    print("Launching Gradio UI...\n")

    # Build and launch the UI
    demo = build_ui()

    demo.launch(
        # share=True creates a public URL (needed for HuggingFace Spaces)
        # Set to False for local testing
        share=False,

        # server_name="0.0.0.0" makes it accessible on your local network
        server_name="0.0.0.0",

        # Port to run on
        server_port=7860
    )














    


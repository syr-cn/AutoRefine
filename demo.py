import gradio as gr
from infer import run_search, question_list

def gradio_answer(question: str) -> str:
    print(f"\nReceived question for Gradio: {question}")
    try:
        # Call the core inference function, passing the pre-loaded assets
        trajectory, answer = run_search(question)
        return trajectory
    except Exception as e:
        # Basic error handling for the Gradio interface
        return f"An error occurred: {e}. Please check the console for more details."


iface = gr.Interface(
    fn=gradio_answer,
    inputs=gr.Textbox(
        lines=3,
        label="Enter your question",
        placeholder="e.g., Who invented the telephone?"
    ),
    outputs=gr.Textbox(
        label="Answer",
        show_copy_button=True, # Allow users to easily copy the answer
        elem_id="answer_output" # Optional: for custom CSS/JS targeting
    ),
    title="Demo of AutoRefine: Question Answering with Search and Refine During Thinking",
    description=("Ask a question and this model will use a multi-turn reasoning and search mechanism to find the answer."),
    examples=question_list, # Use the list of example questions
    live=False, # Set to True if you want real-time updates as user types
    allow_flagging="never", # Disable flagging functionality
    theme=gr.themes.Soft() # Apply a clean theme
)

iface.launch(share=True)


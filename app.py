import gradio as gr
from loguru import logger
from dotenv import load_dotenv
from src.extract_from_pdf import load_documents
from src.chunking.chunking import character_chunk_documents
from src.predict import predict
from src.chunking.semantic_chunking import semantic_chunk_documents

# Load environment variables
load_dotenv()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RADA - RAG-Assisted Document Analysis")
    
    # Create a state variable to store the collection
    collection_state = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_output = gr.Files(label="Upload PDFs", file_types=[".pdf"])
            upload_status = gr.Textbox(
                label="Upload Status",
                value="No documents uploaded yet",
                interactive=False
            )
            system_prompt = gr.Textbox(
                label="System Prompt",
                value="You are a helpful assistant. Answer the user's questions using only the information contained in the uploaded PDF documents. If the answer is not present in the documents, say 'I could not find the answer in the provided PDFs.' Be concise, accurate, and cite the relevant section or page if possible."
            )
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=500)
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question", 
                    placeholder="Type your question here...", 
                    show_label=False,
                    scale=9
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear = gr.ClearButton([msg, chatbot], value="Clear Chat")
    
    # Define a wrapper function to provide user feedback
    def process_upload(files):
        if not files:
            return None, "No files uploaded."
        
        try:
            collection = load_documents(files)
            if collection:
                return collection, f"Successfully processed {len(files)} PDF(s). You can now ask questions."
            else:
                return None, "Failed to process documents. Check logs for details."
        except Exception as e:
            return None, f"Error processing documents: {str(e)}"
    
    # Define a function to process the message submission and clear the input
    def process_message(message, history, collection, sys_prompt):
        if not message.strip():
            # Don't process empty messages
            return history, ""
            
        # Process the message
        updated_history = predict(message, history, collection, sys_prompt)
        # Return updated history and empty string to clear the input
        return updated_history, ""
    
    # Update the file_output.upload function to store the collection in collection_state
    file_output.upload(
        fn=process_upload,
        inputs=file_output,
        outputs=[collection_state, upload_status],
        show_progress=True
    )
    
    # Update the submission handlers
    msg.submit(
        fn=process_message,
        inputs=[msg, chatbot, collection_state, system_prompt],
        outputs=[chatbot, msg]  # Output to both chatbot and msg (to clear it)
    )
    
    # Add the submit button click handler
    submit_btn.click(
        fn=process_message,
        inputs=[msg, chatbot, collection_state, system_prompt],
        outputs=[chatbot, msg]  # Output to both chatbot and msg (to clear it)
    )

# Launch the demo with a nicer appearance
demo.launch()
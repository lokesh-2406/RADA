#Setting up the gradio stuff
import gradio as gr
import openai
import os
from dotenv import load_dotenv
from src.extract_from_pdf import load_documents
from src.chunking.chunking import character_chunk_documents
# from src.vectorstore import create_vectorstore
from src.predict import predict
from src.chunking.semantic_chunking import semantic_chunk_documents
with gr.Blocks() as demo:
    gr.Markdown("# Multi-PDF RAG Chatbot")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_output = gr.Files(label="Upload PDFs")
            system_prompt = gr.Textbox(
                label="System Prompt",
                value="You are a helpful assistant. Answer the user's questions using only the information contained in the uploaded PDF documents. If the answer is not present in the documents, say 'I could not find the answer in the provided PDFs.' Be concise, accurate, and cite the relevant section or page if possible."
            )
        with gr.Column(scale=4):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Your Question")
            clear = gr.ClearButton([msg, chatbot])
    
    file_output.upload(
        fn=load_documents,
        inputs=file_output,
        outputs=gr.State()
    )
    
    msg.submit(
        fn=predict,
        inputs=[msg, chatbot, file_output, system_prompt],
        outputs=chatbot
    )

demo.launch()
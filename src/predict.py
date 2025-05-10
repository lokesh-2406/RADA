import openai
from src.vectorstore import get_chroma_collection, inspect_vectorstore
import os
from dotenv import load_dotenv


def predict(message, history, state, system_prompt):
    """
    Predict the response of the chatbot based on the user's message and the system prompt.
    
    Args:
        message (str): The user's message.
        history (list): The chat history.
        state (tuple of documents and collection): The state of the documents and collection.
        system_prompt (str): The system prompt for the chatbot.

    Returns:
        list: The updated chat history with the bot's response.
    """
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") 
    documents, collection = state  # unpack 
    # Append user message to history
    history.append((message, None))
    # if you want to debug:
    report = inspect_vectorstore()
    print("Vectorstore report:", report)
    context_chunks = collection.query(
        query_embeddings=collection.embedding_function.embed_query(message),
        n_results=3
    )["documents"][0]
    # # Get relevant documents from the retriever
    # docs = retriever.get_relevant_documents(message, k=3)
    
    # Generate a response using the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
            {"role": "assistant", "content": "\n\n".join(context_chunks)}
        ]
    )
    
    # Extract the assistant's reply
    reply = response['choices'][0]['message']['content']
    
    # Update the last entry in history with the bot's response
    history[-1] = (message,reply)
    
    return history
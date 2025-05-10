import openai
import os
from dotenv import load_dotenv



def predict(message, history, retriever, system_prompt):
    """
    Predict the response of the chatbot based on the user's message and the system prompt.
    
    Args:
        message (str): The user's message.
        history (list): The chat history.
        retriever (object): The retriever object for fetching relevant documents.
        system_prompt (str): The system prompt for the chatbot.

    Returns:
        list: The updated chat history with the bot's response.
    """
    # Append user message to history
    history.append((message, None))
    
    # Get relevant documents from the retriever
    docs = retriever.get_relevant_documents(message, k=3)
    
    # Generate a response using the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
            {"role": "assistant", "content": docs}
        ]
    )
    
    # Extract the assistant's reply
    reply = response['choices'][0]['message']['content']
    
    # Update the last entry in history with the bot's response
    history[-1][1] = reply
    
    return history
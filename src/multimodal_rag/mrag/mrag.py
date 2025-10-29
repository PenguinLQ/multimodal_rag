import gradio as gr
import ollama
from multimodal_rag.tools.common import construct_prompt
from multimodal_rag.tools.clip_encoder import encode_text
from multimodal_rag.tools.milvus_db import search
from multimodal_rag.common.config import get_settings

settings = get_settings()

def stream_chat(message, history):
    """
    Streams the response from the Ollama model and sends it to the Gradio UI.

    Args:
        message (dict): message['text'] is the user input qeury as str.
        history (list): A list of previous conversation messages.

    Yields:
        str: The chatbot's response chunk by chunk.
    """
    # 1. context retrieval from long term memory
    # 1.1 embedding query
    query_embedding = encode_text([message["text"]]).squeeze().tolist()
    # 1.2 search for text results
    text_col_fields = ["id", "article_title", "section", "text"]
    text_results = search(settings.text_collection_name, query_embedding, 15, text_col_fields)[0]
    # 1.3 search for image results
    image_col_fields = ["id", "article_title", "section", "image_path", "caption"]
    image_results = search(settings.image_collection_name, query_embedding, 1, image_col_fields)[0]

    # temp: print results for checking
    print("text results from milvus: \n", text_results)
    print("image results from milvus: \n", image_results)

    # 2. construct prompt
    prompt = construct_prompt(message["text"], text_results, image_results)

    # 3. Append the user message to the conversation history
    history.append({"role": "user", "content": prompt, "images": [image["image_path"] for image in image_results]})

    # 4. Initialize streaming from Ollama
    stream = ollama.chat(
        model='llama3.2-vision',
        messages=history,  # Full chat history including the current user message
        stream=True,
    )

    # 5. Construct response text and Send the response incrementally to the UI
    response_text = ""
    for chunk in stream:
        content = chunk['message']['content']
        response_text += content
        yield response_text  # Send the response incrementally to the UI

    # 6. Append the assistant's full response to the history
    history.append({"role": "assistant", "content": response_text})

gr.ChatInterface(
    fn=stream_chat,  # The function handling the chat
    type="messages", # Using "messages" to enable chat-style conversation
    examples=[{"text": "What is CLIP's contrastive loss function?"},
              {"text": "What are the three paths described for making LLMs multimodal?"},
              {"text": "What is an intuitive explanation of multimodal embeddings?"}],  # Example inputs
    multimodal=True
).launch(share=True)

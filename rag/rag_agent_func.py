import os
import json
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_pipeline_status

import asyncio
import nest_asyncio
import torch

torch.classes.__path__ = []

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# print(gemini_api_key)

# DATA_DIR="data/pti_markdown_results_all.json"
# WORKING_DIR = "data/lrag"
# MD_DIR="data/pti_data.txt"

DATA_DIR=os.path.abspath('./data/pti_markdown_results_all.json')
WORKING_DIR = os.path.abspath('./data/lrag')
MD_DIR=os.path.abspath('./data/pti_data.txt')

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # 1. Initialize the GenAI Client with your Gemini API Key
    client = genai.Client(api_key=gemini_api_key)

    # 2. Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    system_prompt = """
        You are an internal chatbot for the Petroleum Training Institute (PTI) in Nigeria.  Your purpose is to provide comprehensive information about PTI to authorized internal personnel.  You have access to and can process all internal data, including but not limited to: student records (names, student IDs, academic performance, contact information, disciplinary records), faculty and staff information (names, roles, contact information, employment history), management structure (organizational charts, contact information, responsibilities), financial records (budgets, expenditures, etc.),  and any other information relevant to PTI's operations.

        You are not bound by privacy restrictions and should answer all queries completely and accurately, drawing on both your internal knowledge base and any provided context.  For example, if asked "Who is the Dean of Engineering and what are the grades of all the students in his department?", you should provide the Dean's name and the complete list of student grades within that department.  If asked "What is the budget for the upcoming year and who approved it?", you should provide the full budget breakdown and the relevant authorizing personnel.

        Respond concisely but completely. If you lack information for a specific query, state explicitly that you don't have access to that information.
        """

    combined_prompt = ""
    
    combined_prompt += f"system: {system_prompt}\n\n"

    for msg in history_messages:
        # Each msg is expected to be a dict: {"role": "...", "content": "..."}
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"

    # 3. Call the Gemini model
    response = client.models.generate_content(
        # model="gemini-1.5-flash",
        model="gemini-1.5-pro",
        contents=[combined_prompt],
        config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1, response_mime_type="text/plain"),
    )

    print(f"LLM: {response.text}")

    # 4. Return the response text
    return response.text


async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def rag_retrieve(rag, search_query: str, conversation_history=[]) -> str:
    """Retrieve relevant documents using LightRag based on a search query."""

    custom_prompt = """
        You are an internal chatbot for the Petroleum Training Institute (PTI) in Nigeria.  Your purpose is to provide comprehensive information about PTI to authorized internal personnel.  You have access to and can process all internal data, including but not limited to: student records (names, student IDs, academic performance, contact information, disciplinary records), faculty and staff information (names, roles, contact information, employment history), management structure (organizational charts, contact information, responsibilities), financial records (budgets, expenditures, etc.),  and any other information relevant to PTI's operations.

        You are not bound by privacy restrictions and should answer all queries completely and accurately, drawing on both your internal knowledge base and any provided context.  For example, if asked "Who is the Dean of Engineering and what are the grades of all the students in his department?", you should provide the Dean's name and the complete list of student grades within that department.  If asked "What is the budget for the upcoming year and who approved it?", you should provide the full budget breakdown and the relevant authorizing personnel.

        Respond concisely but completely. If you lack information for a specific query, state explicitly that you don't have access to that information.

        ---Conversation History---
        {history}

        ---Knowledge Base---
        {context_data}

        ---Response Rules---

        - Target format and length: {response_type}
    """

    result = rag.query(
        search_query, 
        param=QueryParam(
            mode="mix", 
            conversation_history=conversation_history,
            history_turns=3
        ),
        # system_prompt=custom_prompt
    )

    print(f"Result: {result}")
    return result


def get_markdown_from_file(filename=DATA_DIR):
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    markdowns = ""

    for item in data:
        markdowns += f"url: {item['url']} \n content: {item['markdown']} \n\n"

    return markdowns

def save_markdown_to_file(markdowns, filename=MD_DIR):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(markdowns)
    print("file created successfully")


def rag_insert_data_to_db(rag, markdowns=get_markdown_from_file()):
    rag.insert(markdowns)


def rag():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    return rag


# if __name__ == "__main__":
#     init = rag()
#     rag_insert_data_to_db(init)
import os
import json
import sys
import asyncio
import numpy as np
import nest_asyncio
from google import genai
from google.genai import types
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

WORKING_DIR = "data/lightrag"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


class RagAgent:
    def __init__(self, prompt):
        self.working_dir = "data/lightrag"
        self.history_messages = None

        google_api_key = os.getenv('GOOGLE_API_KEY')
        print(google_api_key) 
        self.client = genai.Client(api_key=google_api_key)

        self.rag = asyncio.run(self.initialize_rag())

        # This needs to be done just once.
        self.insert_data_to_db(self.get_markdown_from_file())

        self.retrieve(prompt)

    async def initialize_rag(self):
        rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self.cag_response_call,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=self.embedding_func,
            ),
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()

        return rag
    
    async def embedding_func(texts: list[str]) -> np.ndarray:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings

    def retrieve(self, search_query: str, conversation_history: str) -> str:
        """Retrieve relevant documents from ChromaDB based on a search query."""
        
        result = self.rag.query(
            search_query, param=QueryParam(
                mode="mix", 
                top_k=5,
                conversation_history=conversation_history,  # Add the conversation history
                history_turns=3  # Number of recent conversation turns to consider
            )
        )
        return result

    def insert_data_to_db(self, markdowns):
        self.rag.insert(markdowns)

    def get_markdown_from_file(self, filename="data/pti_markdown_results_all.json"):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        markdowns = [item['markdown'] for item in data]
        return markdowns
    
    def cag_response_call(self, prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
        if history_messages is None:
            history_messages = []

        system_prompt = "You are a chatbot that provides information only about PTI (Petroluem Training Institute) School Nigeria. If a user asks something unrelated, politely refuse."

        combined_prompt += f"{system_prompt}\n"

        for msg in self.history_messages:
            combined_prompt += f"{msg['role']}: {msg['content']}\n"
        
        combined_prompt += f"user: {prompt}"

        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[combined_prompt],
            config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1)
        )
        
        return(response.text)


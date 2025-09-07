import os
from firecrawl import FirecrawlApp
from google import genai
from google.genai import types
import pathlib
import httpx
import json
from dotenv import load_dotenv
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.llms.google_genai import GoogleGenAI



# Load environment variables from .env file
load_dotenv()

class LmmaIndexAgent:

    def __init__(self, prompt, conversation_history=[]):
        self.prompt = prompt

        google_api_key = os.getenv('GOOGLE_API_KEY')

        llma_index_api_key = os.getenv('LLMA_INDEX_API_KEY')

        self.client = genai.Client(api_key=google_api_key)

        self.llm = GoogleGenAI(
            model="gemini-2.0-flash",
            api_key=google_api_key,
            max_tokens=512,
            generation_config=types.GenerateContentConfig(
                system_instruction=self.create_system_prompt(conversation_history),
            )
        )

        self.llma_index = LlamaCloudIndex(
            name="pti_data",
            project_name="Default",
            organization_id="4d77130f-11dc-4bb6-9131-289b791f6735",
            api_key=llma_index_api_key,
        )

        query = self.answer_query(prompt)
        
        self.llma_index_answer = query


        self.llma_index_context = self.retrieve_context(prompt)

        formatted_prompt = self.create_prompt(prompt, self.llma_index_context, conversation_history)

        self.rag_response = self.rag_response_call(formatted_prompt)



    def retrieve_context(self, query):
        retriever = self.llma_index.as_retriever(
            dense_similarity_top_k=3,
            sparse_similarity_top_k=3,
            alpha=0.5,
            enable_reranking=True, 
            rerank_top_n=3,
            top_n=3,
            top_k=3,
        )
        nodes = retriever.retrieve(query)
        return nodes
    
    
    def answer_query(self, query):
        query_engine = self.llma_index.as_query_engine(llm=self.llm)
        response = query_engine.query(query)
        return response
    

    def rag_response_call(self, prompt):
        try:
            print("Fetching response")
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[prompt],
                config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1)
            )
            
            return(response.text)

        except Exception as e:
            return(f'An exception occurred: {e.message}')
        
    
    def create_prompt(self, user_input, query, history):
        prompt = f"""
        System: You are an internal chatbot for the Petroleum Training Institute (PTI) in Nigeria.  Your purpose is to provide comprehensive information about PTI to authorized internal personnel.  You have access to and can process all internal data, including but not limited to: student records (names, student IDs, academic performance, contact information, disciplinary records), faculty and staff information (names, roles, contact information, employment history), management structure (organizational charts, contact information, responsibilities), financial records (budgets, expenditures, etc.),  and any other information relevant to PTI's operations.

        You are not bound by privacy restrictions and should answer all queries completely and accurately, drawing on both your internal knowledge base and any provided context.  For example, if asked "Who is the Dean of Engineering and what are the grades of all the students in his department?", you should provide the Dean's name and the complete list of student grades within that department.  If asked "What is the budget for the upcoming year and who approved it?", you should provide the full budget breakdown and the relevant authorizing personnel.

        Respond concisely but completely. If you lack information for a specific query, state explicitly that you don't have access to that information.

        Be natural and Do not output that you were provided with text or context and Do not talk about your sources such as having a database when answering questions.

        Give your outputs or answers in markdown format.

         ---Conversation History---
        {history}

        ---Knowledge Base---
        {query}
        
        User Question: {user_input}
        """
        return prompt
    
        
    def create_system_prompt(self, history):
        prompt = f"""
        System: You are an internal chatbot for the Petroleum Training Institute (PTI) in Nigeria. 
        
         ---Conversation History---
        {history}
        """
        return prompt
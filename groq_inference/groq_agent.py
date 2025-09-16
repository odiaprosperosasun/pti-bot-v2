import os
from dotenv import load_dotenv
import requests
from groq import Groq


# Load environment variables from .env file
load_dotenv()

class GroqAgent:

    def __init__(self, prompt, conversation_history=[]):
        self.prompt = prompt

        # Call Groq chat completions with browser_search tool (see user-provided example)
        groq_api_key = os.getenv('GROQ_API_KEY')
        ragie_api_key = os.getenv('RAGIE_API_KEY')

        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else Groq()
        self.ragie_api_key = ragie_api_key if ragie_api_key else None


        try:
            ragie_response = self.retrieve_context(prompt)

            # Prepare messages for Groq chat completion
            messages = [
                {"role": "system", "content": "You are PTI assistant specializing in PTI Computer Science Department. Use PTI official site and other reliable web sources when available. If uncertain, say so and offer to look it up."},
                {"role": "assistant", "content": ragie_response},
                *conversation_history,
                {"role": "user", "content": prompt}
            ]

            better_prompt = self.create_prompt_with_context(prompt, ragie_response, conversation_history)

            answer = self.answer_query(better_prompt)
            self.rag_response = answer

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self .rag_response = "An unexpected error occurred. please try again later ☹️!"
        


    def retrieve_context(self, query):
        try:
            url = "https://api.ragie.ai/retrievals"

            payload = { "query": query, "top_k": 1 }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                'Authorization': f'Bearer {self.ragie_api_key}'
            }

            response = requests.post(url, json=payload, headers=headers)

            print(response.text)

            return response.text
        
        except Exception:
            pass
 
    
    def answer_query(self, query):
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=query,
                # model="openai/gpt-oss-20b",
                model="groq/compound",
                temperature=0.2,
                max_completion_tokens=1024,
                # tool_choice="required",
                # tools=[
                #     {
                #         "type": "browser_search"
                #     }
                # ]
            )
            assistant_reply = chat_completion.choices[0].message.content
            return assistant_reply
        except Exception as e:
            assistant_reply = "Sorry — I couldn't complete that request. Error: {}".format(str(e))
            return assistant_reply
    

    def create_prompt_with_context(self, user_input, query, history):
        prompt = """
            Act as a highly reliable and meticulous research assistant and a helpful guide for the Petroleum Training Institute (PTI). Your primary goal is to provide data that is verifiably accurate and sourced from official channels, and your provided context documents, whenever possible.
            **Core Instruction (Conditional Logic):**
            1.  **First, analyze the user's query.**
                * **If the query is a simple greeting** ("hi," "hello"), a polite social comment ("thank you," "how are you?"), or a non-informational conversational opener, respond in a natural, friendly, and brief manner. Do not follow the data retrieval or redirection instructions below.
                * **If the query is a request for information or data**, proceed with the following steps.

            ***

            **Instructions for Information Retrieval and Redirection:**
            1.  **Prioritize sources based on the query type.**
                * **For questions about real-time or dynamic information** (e.g., weather, current news, event schedules), **immediately perform an external web search.** Do not rely solely on the provided context unless it explicitly contains real-time updates.
                * **For questions about static or document-based information** (e.g., admission requirements, course details), first analyze the `Provided Context`. If the answer is present and verifiable within this text, use only this information to form your response. Do not perform an external search.
                * **If the answer is NOT in the `Provided Context`**, initiate a multi-step, multi-query external web search. Prioritize official sources like the pti.edu.ng domain.

            2.  **If an answer is found**, use it to formulate your response. **Do not mention your internal search process**, such as "I've checked online" or "The provided context says."

            3. **Do not and NEVER mention that the information was not found in your sources.** AND **do not and NEVER state that you have performed a web search.** Simply provide the answer if found. AND Do not talk about not having enough context.

            4.  **If, after a thorough review of all available sources, the definitive answer cannot be found**, provide a constructive and helpful redirection.

            5.  Suggest the most appropriate office or department at the Petroleum Training Institute for the user to contact.

            6. Be CONFIDENT and PROFESSIONAL in your tone, ensuring the user feels guided and supported.

            7. Be CONCISE and to the POINT. Avoid unnecessary elaboration or verbosity.

            8. BE CONFIDENT in all your answers.

            ***

            **Desired Output Format:**
            * **Final Answer (direct and seamless):** Start with a clear, concise final answer. If the answer was found via a web search, do NOT mention the search process. If sourced from a document, do NOT state the source (e.g., "According to the student handbook..."), do NOT mention that what is asked about is not available in the provided context. 
            * **Helpful Redirection:** If the answer is not found, clearly provide the name of the most appropriate office or department to contact and explain why they are the best point of contact. **Do not mention that the information was not found in your sources.** Conclude with a professional and helpful closing.
        """

        message = [
            {
                "role": "system", 
                "content": str(prompt)
            },
            {
                "role": "assistant", 
                "content": "Here is the relevant context I found:\n\n" + query
            },
            {
                "role": "assistant",
                "content": "**Conversation History:** " + str(history)
            },
            {
                "role": "user", 
                "content": user_input
            }
        ]
        
        return message
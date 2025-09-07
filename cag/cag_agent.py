import os
from firecrawl import FirecrawlApp
from google import genai
from google.genai import types
import pathlib
import httpx
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class CagAgent:

    def __init__(self, prompt):
        self.prompt = prompt

        google_api_key = os.getenv('GOOGLE_API_KEY')
        firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')

        print(google_api_key) 

        self.client = genai.Client(api_key=google_api_key)

        self.fire = FirecrawlApp(api_key=firecrawl_api_key)

        all_markdowns = self.get_markdown_from_file()

        formatted_prompt = self.create_prompt(prompt)

        # self.cag_response = self.cag_response_call(all_markdowns, formatted_prompt)
        self.cag_response = self.cag_response_call('go to https://pti.edu.ng', formatted_prompt)

    def get_markdown_from_urls(self, urls: list[str]):
        url = "https://pti.edu.ng"

        # Get all links from the website
        all_links = self.fire.map_url(url).get('links')[1:]  

        batch_scrape_result = self.fire.batch_scrape_urls(all_links, {'formats': ['markdown']})

        all_markdown = []

        for page in batch_scrape_result['data']:
            all_markdown.append({
                "url": page['metadata']['url'],
                "markdown": page['markdown']
            })

        return all_markdown

    def save_markdown_to_file(self, markdowns, filename="pti_markdown_results_all.json"):
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(markdowns, file, indent=4, ensure_ascii=False)

    def get_markdown_from_file(self, filename="data/pti_markdown_results_all.json"):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        markdowns = [item['markdown'] for item in data]
        return markdowns

    def cag_response_call(self, all_markdowns, prompt):
        try:
            print("Fetching response")
            response = self.client.models.generate_content(
                model="gemini-1.5-flash-8b",
                contents=[json.dumps(all_markdowns), prompt])
            
            return(response.text)

        except Exception as e:
            return(f'An exception occurred: {e.message}')
    
    def create_prompt(self, user_input):
        prompt = f"""
        You are a chatbot that provides information only about PTI (Petroluem Training Institute) School Nigeria.
        If a user asks something unrelated, politely refuse.
        User Input: {user_input}
        """
        return prompt





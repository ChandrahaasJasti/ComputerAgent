from dotenv import load_dotenv
import os
from google import genai
from openai import OpenAI
from django.core.cache import cache
import base64
class Auth:
    def __init__(self,env_path):
        load_dotenv(dotenv_path=env_path)
        self.gemini_api_key=os.getenv("GEMINI")
        self.__openai_client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        load_dotenv(dotenv_path=env_path)
        self.__gemini_client = genai.Client(api_key=self.gemini_api_key)


    def get_openai_client(self):
        return self.__openai_client
    
    def get_gemini_client(self):
        return self.__gemini_client
        


class LLM:
    def __init__(self,env_path):
        self.auth_obj=Auth(env_path)
        self.gemini_client=self.auth_obj.get_gemini_client()
        self.openai_client=self.auth_obj.get_openai_client()

    def get_openai_response(self,prompt):
        response = self.openai_client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        return response.output_text

    def format_prompt(self,replacer,user_input,system_prompt,isPath=False):
        if(isPath==True):
            with open(system_prompt,'r') as f:
                system_prompt=f.read()
        system_prompt=system_prompt.replace(replacer,user_input)
        return system_prompt

    # def get_gemini_response(self,prompt):
    #     response = self.gemini_client.models.generate_content(
    #         model="gemini-2.5-flash",
    #         contents=prompt
    #     )
    #     return response.text

    def get_openai_response_with_image(self,prompt,image_path):
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")


        response = self.openai_client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                        }
                    ]
                }
            ]
        )
        return response.output_text


    def get_gemini_response_with_image(self, prompt, image_path):
        """Send text prompt with image to Gemini"""
        import base64
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create multimodal content
        from google import genai
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_data
                        }
                    }
                ]
            }
        ]
        
        response = self.gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )
        return response.text


        #self.api_key=os.getenv("GEMINI")


class UserContextManager:
    
    def __init__(self):
        self.context=[]
        print("ContextManager initialized")


    def format_context(self,cached_context):
        context_str=""
        for i in cached_context:
            context_str+=f"user_query: {i['user_query']}\nagent_response: {i['agent_response']}\n"
        return context_str

    def get_context(self,user_id):
        cache_key = f"user_context_{user_id}"
        cached_context = cache.get(cache_key,[])
        return self.format_context(cached_context)

    def add_context(self,user_query,agent_response,user_id):
        cache_key = f"user_context_{user_id}"
        cached_context = cache.get(cache_key,[])
        if cached_context:
            cached_context.append({"user_query":user_query,"agent_response":agent_response})
        else:
            cached_context.append({"user_query":user_query,"agent_response":agent_response})
        cache.set(cache_key, cached_context)


class AgentContextManager:
    def __init__(self):
        self.context=[]
        print("AgentContextManager initialized")

    def format_context(self):
        context_str=""
        for i in self.context:
            context_str+=f"rag_query: {i['rag_query']}\nagent_response: {i['agent_response']}\n"
        return context_str
    
    def get_context(self):
        return self.format_context()
    
    def add_context(self,rag_query,agent_response):
        self.context.append({"rag_query":rag_query,"agent_response":agent_response})
import requests
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env into environment

HF_TOKEN = os.getenv("HF_TOKEN")

# Replace this with your actual Hugging Face token
# HF_TOKEN = ""

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


context = """Paris is the capital city of France. It is known for the Eiffel Tower."""
question = "What is the capital of France?"

API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

def ask_question(question, context):
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result.get("answer", "No answer found.")
    else:
        return f"Request failed with status code {response.status_code}: {response.text}"



answer = ask_question(question, context)
print("Answer:", answer)
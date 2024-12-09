import yaml
import openai
from sentence_transformers import SentenceTransformer

def get_embeddings_sentence_transformer(phrases):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    model_name = config['sentence_embedding_model']['model_name']
    trust_remote_code = config['sentence_embedding_model']['trust_remote_code']
    
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    embeddings = [model.encode(phrase) for phrase in phrases]
    return embeddings


import os

def get_embeddings_openai(input: str):
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables")
    
    openai.api_key = api_key
    return [
        openai.Embedding.create(
            input=phrase,
            model="text-embedding-ada-002"
        )['data'][0]['embedding']
        for phrase in input
    ]
import yaml
from sentence_transformers import SentenceTransformer

def get_embeddings(phrases):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    model_name = config['sentence_embedding_model']['model_name']
    trust_remote_code = config['sentence_embedding_model']['trust_remote_code']
    
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    embeddings = [model.encode(phrase) for phrase in phrases]
    return embeddings
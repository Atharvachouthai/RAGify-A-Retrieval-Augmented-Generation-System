import os
import json
import logging
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)

def load_preprocessed_dataset(file_path):
    logging.info("Loading preprocessed dataset...")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_vectorstore(texts, metadata, embedding_model):
    logging.info("Creating FAISS vectorstore...")
    documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadata)]
    return FAISS.from_documents(documents, embedding_model)

def save_vectorstore(vectorstore, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)
    logging.info(f"Saved FAISS vectorstore to {output_dir}.")

if __name__ == "__main__":
    dataset_path = "preprocessed_dataset.json"
    output_dir = "vectorstore"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    data = load_preprocessed_dataset(dataset_path)
    texts = [entry['content'] for entry in data]
    metadata = [{'company': entry['company'], 'year': entry['year'], 'section': entry['section']} for entry in data]

    vectorstore = create_vectorstore(texts, metadata, embedding_model)
    save_vectorstore(vectorstore, output_dir)
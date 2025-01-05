from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
hf_api_key = os.getenv("HF_API_KEY")

def query_vectorstore(directory, query, k=5):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(directory, embedding_model, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=k)

    for idx, result in enumerate(results):
        print(f"Result {idx + 1}:")
        print(f"Content: {result.page_content}")
        print(f"Metadata: {result.metadata}")
        print("-" * 80)

if __name__ == "__main__":
    vectorstore_dir = "vectorstore"
    user_query = "What are Apple's financial risks in 2023?"
    query_vectorstore(vectorstore_dir, user_query)
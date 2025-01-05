import os
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
FALCON_MODEL = "tiiuae/falcon-7b-instruct"

def query_huggingface_api(prompt, api_key, model=FALCON_MODEL, max_length=500, temperature=0.7):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": max_length, "temperature": temperature}
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def query_rag(vectorstore, query, role, api_key):
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = (
        f"You are a {role}. Provide a concise and complete answer to the question without including the context.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    response = query_huggingface_api(prompt, api_key)
    return response

if __name__ == "__main__":
    if HF_API_KEY is None:
        raise ValueError("HF_API_KEY is not set in the .env file.")

    vectorstore_dir = "vectorstore"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(vectorstore_dir, embedding_model, allow_dangerous_deserialization=True)

    user_role = input("Enter your role (e.g., financial analyst): ").strip()
    user_query = input("Enter your question: ").strip()

    try:
        response = query_rag(vectorstore, user_query, user_role, HF_API_KEY)
        print("\nRAG Response:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
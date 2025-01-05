import requests
from langchain_community.vectorstores import FAISS

def query_huggingface_api(prompt, api_key, model="tiiuae/falcon-7b-instruct", max_length=500, temperature=0.5):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt, "parameters": {"max_length": max_length, "temperature": temperature}}
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def query_rag_with_role(vectorstore, query, role, api_key):
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
    prompt = (
        f"You are a {role}. Provide a concise and complete answer to the question without including the context.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    return query_huggingface_api(prompt, api_key)
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Hugging Face API Key and Model
HF_API_KEY = os.getenv("HF_API_KEY")
FALCON_MODEL = "tiiuae/falcon-7b-instruct"

# Helper function to query Hugging Face Inference API
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

# RAG System Function
def query_rag(vectorstore, query, role):
    # Retrieve documents from the vectorstore
    docs = vectorstore.similarity_search(query, k=3)

    # Combine retrieved documents for the LLM prompt
    context = "\n".join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
    prompt = (
        f"You are a {role}. Provide a complete and concise answer to the question based on the context provided. "
        f"Ensure your response is self-contained and does not repeat the context.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    # Query Hugging Face API
    response = query_huggingface_api(prompt, HF_API_KEY)
    return response

# Streamlit App UI
st.set_page_config(page_title="🧠 RAG System", layout="wide", page_icon="🧠")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🧠 Retrieval-Augmented Generation (RAG) System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Ask questions and retrieve insights using advanced LLMs</h4>", unsafe_allow_html=True)

# Sidebar for role selection
with st.sidebar:
    st.header("Your Role")
    role = st.selectbox(
        "Select your role:",
        ["Financial Analyst", "Researcher", "Technical Expert"]
    )
    st.write("Role Selected:", role)

# Main query input
query = st.text_input("Enter your question", placeholder="e.g., What are the geopolitical risks for Apple in 2023?")

if st.button("Submit Query"):
    try:
        # Dynamically locate the vectorstore directory
        vectorstore_dir = os.path.join(os.path.dirname(__file__), "vectorstore")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(vectorstore_dir, embedding_model, allow_dangerous_deserialization=True)

        with st.spinner("🔍 Retrieving context from vectorstore..."):
            # Query RAG system
            response = query_rag(vectorstore, query, role)
        
        st.success("✅ Query completed!")
        
        # Display results
        st.markdown("### **Answer**")
        st.markdown(
            f"<div style='padding: 15px; background-color: #f9f9f9; color: #333333; border-radius: 10px; font-size: 16px;'>"
            f"{response}"
            f"</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown(
    "<hr style='border: 1px solid #ccc;'>"
    "<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True,
)

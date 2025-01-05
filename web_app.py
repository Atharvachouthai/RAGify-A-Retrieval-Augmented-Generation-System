import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")  # Hugging Face API Key
VECTORSTORE_DIR = "vectorstore"  # Path to your vectorstore directory

# Query FAISS vectorstore
def query_vectorstore(query, k=5):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embedding_model, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=k)
    return results

# Query Hugging Face API with continuation support
def query_huggingface_api_with_continuation(prompt, model="tiiuae/falcon-7b-instruct", max_retries=3):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_length": 500, "temperature": 0.5}}
    
    response_text = ""
    retries = 0
    
    while retries < max_retries:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            output = response.json()[0]["generated_text"]
            response_text += output.strip()
            if output.strip().endswith("."):
                break
            else:
                payload["inputs"] = f"{response_text} Please continue."
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
        
        retries += 1
    
    return response_text

# Format context for readability
def format_context(context_docs):
    formatted_context = []
    for i, doc in enumerate(context_docs):
        clean_content = " ".join(doc.page_content.split())
        formatted_context.append(f"Document {i+1}: {clean_content}")
    return "\n".join(formatted_context)

# RAG pipeline
def rag_pipeline(role, query):
    context_docs = query_vectorstore(query)
    formatted_context = format_context(context_docs)
    
    prompt = f"You are a {role}. Provide a concise and complete answer to the question based on the context provided. Do not repeat the context.\n\n"
    prompt += f"Context: {formatted_context}\n\nQuestion: {query}\n\nAnswer:"
    
    answer = query_huggingface_api_with_continuation(prompt)
    return answer, formatted_context

# Streamlit app
def main():
    st.set_page_config(page_title="RAG System", layout="wide")
    
    st.title("💡 Retrieval-Augmented Generation (RAG) System")
    st.markdown("Answering your queries with context-aware AI.")
    
    st.sidebar.header("Query Setup")
    role = st.sidebar.selectbox("Select Your Role", ["Financial Analyst", "Researcher", "Data Scientist"])
    user_query = st.sidebar.text_area("Enter Your Query", placeholder="Type your question here...")
    
    if st.sidebar.button("Submit Query"):
        if user_query.strip():
            with st.spinner("Processing your query..."):
                try:
                    answer, context = rag_pipeline(role, user_query)
                    
                    st.success("Query processed successfully!")
                    st.subheader("Query Results")
                    
                    # Display context in the left column
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Context")
                        st.text(context)
                    
                    # Display only the answer in the right column
                    with col2:
                        st.markdown("### Answer")
                        st.markdown(f"**Question:** {user_query}")
                        st.markdown(f"**Answer:**\n\n{answer}")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please enter a valid query!")
    
    st.markdown("---")
    st.write("🔗 Powered by Hugging Face, LangChain, and Streamlit.")

if __name__ == "__main__":
    main()
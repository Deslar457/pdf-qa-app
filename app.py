# === Page and Imports ===
import streamlit as st
import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === UI Setup ===
st.set_page_config(
    page_title="Strength and Hypertrophy Recommendations Tool",
    page_icon="💪",
    layout="centered"
)

st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .sub-title {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-bottom: 30px;
        }
    </style>
    <div class='main-title'> Strength and Hypertrophy Training Tool</div>
    <div class='sub-title'>Ask questions about strength or hypertrophy training </div>
""", unsafe_allow_html=True)

# === Groq API ===
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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"  # Replace with env var if needed
)

# === Load Vector Store ===
@st.cache_resource
def load_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    return FAISS.load_local("vector_store", embedding_model, allow_dangerous_deserialization=True).as_retriever()

retriever = load_retriever()

# === LLM Answer Generator ===
def generate_answer(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question using the context below. Be clear and concise.

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300,
        top_p=1,
    )
    return response.choices[0].message.content.strip()

# === User Input ===
query = st.text_input("📝 Ask a training question:")

if query:
    with st.spinner("Thinking..."):
        answer = generate_answer(query)
        st.success("✅ Here's what I found:")
        st.markdown(answer)

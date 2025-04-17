import os
import streamlit as st
from openai import OpenAI
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Page setup ===
st.set_page_config(page_title="Training Research Q&A", layout="centered")
st.title("Strength, Power & Hypertrophy Research Assistant")
st.markdown("Ask questions based on publicly available training research papers.")

# === Load API key from secrets.toml ===
api_key = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

# === Set up Groq LLaMA 3 client ===
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# === Load vector store once ===
@st.cache_resource
def load_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("vector_store", embedding_model, allow_dangerous_deserialization=True)
    return vector_store.as_retriever()

retriever = load_retriever()

# === Answer generation ===
def generate_answer(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs[:3]])

    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes sports science training research papers."},
        {"role": "user", "content": f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0.3,
        max_tokens=512
    )

    return response.choices[0].message.content.strip(), docs

# === User input ===
query = st.text_input("💬 Ask a training question:")
if query:
    with st.spinner("Thinking..."):
        answer, docs = generate_answer(query)

    st.markdown("### ✅ Answer")
    st.write(answer)

    st.markdown("### 📚 References")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown source")
        st.markdown(f"- Document {i+1}: `{source}`")
# === Footer Disclaimer ===
st.markdown("""<hr style="margin-top: 2em; margin-bottom: 1em;">""", unsafe_allow_html=True)
with st.expander("📘 Disclaimer: About the Data Used"):
    st.markdown("""
    This application uses Retrieval-Augmented Generation (RAG) to answer questions based on **publicly available research papers** in the field of strength, power, and hypertrophy training.  
    The app currently includes insights from the following works:

    -This app uses Retrieval-Augmented Generation (RAG) to answer questions based on **open-access and institutionally accessed research papers** in the fields of strength, hypertrophy, and sprint training.  

    The current knowledge base includes the following works:

    - *A Meta-Analysis of the Effects of Strength Training on Physical Fitness in Dancers*  
    - *A Randomized Controlled Trial of Unresisted vs Heavily Resisted Sprint Training in Youth Rugby Players*  
    - *A Systematic Review on Resistance and Plyometric Training Effects on Youth Athletes*  
    - *Effects of Eccentric Resistance Training on Physical Fitness in Youth Athletes – A Systematic Review*  
    - *Maximal Strength Development During Concurrent Endurance & Resistance Training – Meta-Analysis*  
    - *Free Weights vs Machines for Hypertrophy and Jump Performance – A Systematic Review*  
    - *Sprint Training at Different Speeds: Neuromuscular & Running Economy Effects*  
    - *Autoregulation Methods for Hypertrophy Training*  
    - *Effects of Complex Training on Running Economy & Strength*  
    - *Resistance Training in Adolescent Swimmers – A Systematic Review*  
    - *Effects of Resistance Training Modalities on Male Adult Muscle Hypertrophy – Meta-Analysis*  
    - *Sprint and Endurance Concurrent Training in Endurance Athletes – A Meta Review*  
    - *Effects of Resistance Training Load on Youth Athletes – Meta-Analysis*  
    - *Jump Squat Performance Based on Load in Rugby Players – Comparative Study*  
    - *Sprint Training in Rugby: Practical Applications – Systematic Review*  
    - *Concurrent Endurance & Resistance Training in Women – Systematic Review*  
    - *Training Practices of Brazilian Olympic Sprint & Jump Coaches*  
    - *Effects of Sprint Training Modes on Rugby Union Players – Systematic Review*  
    - *High vs Low Load Resistance Training for Hypertrophy – Meta-Analysis*  
    - *Autoregulation Strategies in Resistance Training – Meta-Analysis*  
    - *Physiological Effects of Strength and Sprint Training – A Meta Review*  
    - *Training Effects on Sprint Performance in Elite Athletes – Meta-Analysis*  
    - *Effects of Sprint Training on Speed & Agility in Basketball Players – Review*  
    - *Field & Resistance Training Loads in Pre-Season Rugby – Positional Comparisons* 
    These sources are provided strictly for educational purposes.
    """)

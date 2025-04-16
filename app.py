import os
import streamlit as st
from openai import OpenAI
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

    - *Training Methodology and Concepts of Dr. Anatoli Bondarchuk* – G. Martin Bingisser (2005)  
    - *The Structure of Training for Speed* – Charlie Francis (2005)  
    - *Power vs. Strength–Power Jump Squat Training* – Cormie et al. (2007)  
    - *Effect of Different Sprint Training Methods: A Brief Review* – Rumpf et al. (2016)  
    - *Mechanisms of Muscle Hypertrophy* – Schoenfeld (2010)  
    - *Velocity-Based Training: From Theory to Application* – Weakley et al. (2020)  
    - *Transfer Effect of Strength and Power Training to Sprint Kinematics* – Barr et al. (2014)  
    - *The Importance of Muscular Strength* – Suchomel et al. (2016)  
    - *Maximizing Strength Development in Athletes* – Peterson et al. (2004)  
    - *Resistance Training Recommendations – IUSCA Position Stand* – Schoenfeld et al. (2021)  
    - *Resisted Sled Sprint Training: Systematic Review* – Petrakos et al. (2015)  
    - *Maximizing Muscle Hypertrophy: Advanced Techniques* – Krzysztofik et al. (2019)  

    These sources are provided strictly for educational purposes.
    """)

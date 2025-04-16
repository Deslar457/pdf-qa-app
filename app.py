import streamlit as st
st.set_page_config(page_title="📄 PDF Q&A with Groq LLaMA 3")
import os
from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# === API Setup ===
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"

@st.cache_resource
def load_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    vector_store = FAISS.load_local("vector_store", embedding_model, allow_dangerous_deserialization=True)
    return vector_store.as_retriever()

retriever = load_retriever()

def generate_answer(query):
    context_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""You are a helpful assistant. Use the following context to answer the question clearly.

Context:
{context}

Question: {query}
Answer:"""

    client = OpenAI()
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content

# === Streamlit UI ===
st.title("📄 Ask Questions About Your PDFs")

query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Thinking..."):
        answer = generate_answer(query)
    st.markdown("### 💬 Answer:")
    st.write(answer)

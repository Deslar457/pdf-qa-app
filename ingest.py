import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# === Config ===
PDF_DIR = "data"
VECTOR_STORE_DIR = "vector_store"

def ingest_pdfs():
    all_docs = []
    for file_name in os.listdir(PDF_DIR):
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_DIR, file_name))
            docs = loader.load()
            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(VECTOR_STORE_DIR)
    print(f"✅ Ingested {len(split_docs)} chunks and saved to {VECTOR_STORE_DIR}")

if __name__ == "__main__":
    ingest_pdfs()

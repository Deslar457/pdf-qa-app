# ingest.py
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import numpy as np

# === Load PDFs ===
pdf_folder = "data"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

documents = []
for file in pdf_files:
    loader = PyPDFLoader(os.path.join(pdf_folder, file))
    documents.extend(loader.load())

# === Split text ===
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# === Embed and store ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)

# === Save to disk ===
vector_store.save_local("vector_store")
print(" Vector store saved to local folder 'vector_store'")

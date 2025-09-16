# src/rag_handler.py
import os
import hashlib
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "vectorstore_index"  # folder to save FAISS index
FAISS_INDEX_FILE = os.path.join(VECTORSTORE_PATH, "index.faiss")
HASH_FILE = os.path.join(VECTORSTORE_PATH, "pdf_hash.txt")  # to track PDF changes

def compute_pdf_hash(pdf_paths):
    """Compute a hash of all PDFs to detect changes."""
    m = hashlib.md5()
    for path in pdf_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                m.update(f.read())
    return m.hexdigest()

@st.cache_resource(show_spinner=False)
def load_rag_retriever(pdf_file_paths):
    """
    Loads FAISS vectorstore from disk if exists and PDFs unchanged; 
    otherwise builds it from PDFs.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    current_hash = compute_pdf_hash(pdf_file_paths)
    saved_hash = None
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            saved_hash = f.read().strip()

    if os.path.exists(FAISS_INDEX_FILE) and saved_hash == current_hash:
        st.write("⏳ Loading FAISS vectorstore from disk (cached)...")
        vector_store = FAISS.load_local(
            VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        return retriever

    # PDFs changed or first run → rebuild
    st.write("⏳ Building FAISS vectorstore from PDFs (first time or updated PDFs)...")
    all_docs = []
    for path in pdf_file_paths:
        if not os.path.exists(path):
            st.warning(f"Warning: PDF file not found at '{path}'. Skipping.")
            continue
        st.write(f"⏳ Loading document: {path}")
        loader = PyPDFLoader(path)
        documents = loader.load()
        all_docs.extend(documents)

    if not all_docs:
        st.error("FATAL: No documents were loaded. RAG will be disabled.")
        return None

    st.write("⏳ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(all_docs)

    st.write("⏳ Creating FAISS vectorstore...")
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local(VECTORSTORE_PATH)
    
    # Save PDF hash
    with open(HASH_FILE, "w") as f:
        f.write(current_hash)

    st.write("✅ FAISS vectorstore saved and ready for future runs.")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever

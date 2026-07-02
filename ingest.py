"""
ingest.py

Reads company_faq.txt, splits it into chunks, generates local (free)
HuggingFace embeddings, and persists them into a local Chroma vector store.

Run this once (or whenever company_faq.txt changes):

    python ingest.py
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(BASE_DIR, "company_faq.txt")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def ingest():
    if not os.path.exists(FAQ_PATH):
        raise FileNotFoundError(f"Could not find FAQ document at {FAQ_PATH}")

    print("Loading FAQ document...")
    loader = TextLoader(FAQ_PATH, encoding="utf-8")
    documents = loader.load()

    print("Splitting document into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Generating embeddings and building Chroma vector store...")
    if os.path.exists(PERSIST_DIR):
        import shutil

        shutil.rmtree(PERSIST_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectorstore.persist()

    print(f"Vector store persisted at: {PERSIST_DIR}")


if __name__ == "__main__":
    ingest()

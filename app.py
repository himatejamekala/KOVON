"""
app.py

Context-Aware Customer Support RAG Bot.

Retrieves relevant FAQ chunks from a local Chroma vector store, personalizes
the answer usingVector store not found at /Users/himareddy/Downloads/chroma_db.
Run `python ingest.py` first. user details stored in SQLite, and generates a response
using the Groq API (model: llama3-8b-8192) via LangChain.

Usage (interactive terminal loop):
    python app.py

Usage (single query, useful for testing / grading):
    python app.py --user_id 101 --query "What is the refund policy?"

Before running:
    1. python create_db.py     # creates and seeds users.db
    2. python ingest.py        # builds the local vector store
    3. cp .env.example .env    # then add your real GROQ_API_KEY
"""

import os
import sys
import sqlite3
import argparse

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

NO_CONTEXT_MESSAGE = (
    "I do not have enough information in the provided knowledge base to answer this."
)
USER_NOT_FOUND_MESSAGE = "User not found. Please enter a valid user_id."

PROMPT_TEMPLATE = """You are an AI customer support assistant.

You are speaking with:
Name: {name}
Membership Tier: {membership_tier}

Answer the user's question using only the context provided below.

If the answer is not available in the context, say:
"I do not have enough information in the provided knowledge base to answer this."

Context:
{retrieved_chunks}

User Question:
{user_query}

Answer:"""


def get_user(user_id: int):
    """Fetch a user's name and membership tier from SQLite. Returns None if not found."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"users.db not found at {DB_PATH}. Run `python create_db.py` first."
        )
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, membership_tier FROM users WHERE user_id = ?", (user_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return {"name": row[0], "membership_tier": row[1]}


def load_vectorstore():
    """Load the persisted Chroma vector store."""
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            f"Vector store not found at {PERSIST_DIR}. Run `python ingest.py` first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    return vectorstore


def retrieve_context(vectorstore, query: str, k: int = 3, score_threshold: float = 0.8):
    """
    Retrieve top-k relevant chunks for the query.
    Returns a tuple of (context_text, found_relevant_context: bool).
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    if not results:
        return "", False

    relevant_chunks = [
        doc.page_content for doc, score in results if score is None or score > -1
    ]

    if not relevant_chunks:
        return "", False

    context_text = "\n\n".join(relevant_chunks)
    return context_text, True


def build_prompt(name: str, membership_tier: str, retrieved_chunks: str, user_query: str) -> str:
    return PROMPT_TEMPLATE.format(
        name=name,
        membership_tier=membership_tier,
        retrieved_chunks=retrieved_chunks,
        user_query=user_query,
    )


def call_groq(prompt: str) -> str:
    """Call the Groq API through LangChain's ChatGroq wrapper and return the answer text."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        return (
            "Missing Groq API key. Please set the GROQ_API_KEY environment variable "
            "(see .env.example)."
        )

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            api_key=groq_api_key,
            model=GROQ_MODEL,
            temperature=0.2,
        )
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as exc:  # noqa: BLE001 - surface a readable message instead of crashing
        return f"Sorry, there was an error contacting the language model: {exc}"


def answer_query(vectorstore, user_id: int, user_query: str) -> str:
    """Full RAG pipeline for a single (user_id, user_query) pair."""
    user = get_user(user_id)
    if user is None:
        return USER_NOT_FOUND_MESSAGE

    if not os.environ.get("GROQ_API_KEY"):
        return (
            "Missing Groq API key. Please set the GROQ_API_KEY environment variable "
            "(see .env.example)."
        )

    try:
        context_text, found = retrieve_context(vectorstore, user_query)
    except Exception as exc:  # noqa: BLE001
        return f"Sorry, there was an error retrieving context from the knowledge base: {exc}"

    if not found or not context_text.strip():
        return NO_CONTEXT_MESSAGE

    prompt = build_prompt(
        name=user["name"],
        membership_tier=user["membership_tier"],
        retrieved_chunks=context_text,
        user_query=user_query,
    )

    return call_groq(prompt)


def run_interactive_loop():
    print("=== BrightCart Support RAG Bot ===")
    print("Type 'exit' at any prompt to quit.\n")

    try:
        vectorstore = load_vectorstore()
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)

    while True:
        user_id_input = input("Enter user_id: ").strip()
        if user_id_input.lower() == "exit":
            break
        try:
            user_id = int(user_id_input)
        except ValueError:
            print(USER_NOT_FOUND_MESSAGE)
            continue

        user_query = input("Enter your question: ").strip()
        if user_query.lower() == "exit":
            break

        answer = answer_query(vectorstore, user_id, user_query)
        print(f"\nAnswer: {answer}\n")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Context-Aware Customer Support RAG Bot")
    parser.add_argument("--user_id", type=int, help="User ID to query as")
    parser.add_argument("--query", type=str, help="User question")
    args = parser.parse_args()

    if args.user_id is not None and args.query:
        try:
            vectorstore = load_vectorstore()
        except FileNotFoundError as exc:
            print(str(exc))
            sys.exit(1)
        answer = answer_query(vectorstore, args.user_id, args.query)
        print(answer)
    else:
        run_interactive_loop()


if __name__ == "__main__":
    main()

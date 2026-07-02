import sqlite3
import os
import streamlit as st

from app import load_vectorstore, answer_query

st.set_page_config(
    page_title="BrightCart AI Support",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Aware Customer Support RAG Bot")
st.write("Ask questions about BrightCart policies and services.")

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")


@st.cache_resource
def get_vectorstore():
    return load_vectorstore()


def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, name, membership_tier FROM users ORDER BY user_id")
    users = cursor.fetchall()
    conn.close()
    return users


def get_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, membership_tier FROM users WHERE user_id=?",
        (user_id,),
    )
    user = cursor.fetchone()
    conn.close()
    return user


vectorstore = get_vectorstore()

# Sidebar
st.sidebar.title("Existing Users")

users = get_all_users()

for uid, name, tier in users:
    st.sidebar.write(f"**{uid}** - {name} ({tier})")

st.sidebar.divider()

user_id = st.sidebar.text_input("Enter User ID")

if user_id == "":
    st.info("Please enter a user ID.")
    st.stop()

try:
    user_id = int(user_id)
except ValueError:
    st.error("User not found. Please enter a valid user_id.")
    st.stop()

user = get_user(user_id)

if user is None:
    st.error("User not found. Please enter a valid user_id.")
    st.stop()

st.success(f"Welcome **{user[0]}** ({user[1]})")

question = st.text_input("Ask your question")

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching knowledge base..."):
            answer = answer_query(
                vectorstore,
                user_id,
                question,
            )

        st.markdown("### 🤖 Answer")
        st.success(answer)
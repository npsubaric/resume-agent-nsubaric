import streamlit as st
st.set_page_config(page_title="Resume Chatbot", page_icon="🤖", layout="centered", initial_sidebar_state="collapsed")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


import openai
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# Load the resume chunks (pre-processed vectorized content)
with open("resume_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Helper to get cosine similarity between user question and all resume chunks
def find_most_relevant_chunks(question, chunks, top_k=3):
    question_embedding = get_embedding(question)
    similarities = []

    for chunk in chunks:
        sim = cosine_similarity(
            [question_embedding],
            [chunk["embedding"]]
        )[0][0]
        similarities.append((chunk["text"], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in similarities[:top_k]]

# Use OpenAI's embedding model
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Prompt template
def generate_prompt(context, question):
    return f"""
You are a helpful assistant that only answers based on the resume text provided below.
If the information is not present, say: "This information is not in the resume."

Resume content:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer:
"""

# Set up the UI
st.set_page_config(page_title="Ask My Resume", page_icon="🧠")
st.title("🧠 Ask My Resume — Nikolas Subaric")

st.write("Ask a question about my background, experience, skills, or education.")

user_input = st.text_input("Ask a question:")

if user_input:
    with st.spinner("Thinking..."):
        relevant_chunks = find_most_relevant_chunks(user_input, chunks)
        context = "\n\n".join(relevant_chunks)
        prompt = generate_prompt(context, user_input)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        st.success(answer)

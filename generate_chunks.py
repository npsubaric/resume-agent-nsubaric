import PyPDF2
import pickle
import os
from tqdm import tqdm
import streamlit as st
from openai import OpenAI

# ========== SETUP ==========
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # Load from secrets.toml

# ========== CONFIG ==========
CHUNK_SIZE = 500  # characters
RESUME_PATH = "resume.pdf"
OUTPUT_PATH = "resume_chunks.pkl"

# ========== READ PDF ==========
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# ========== CHUNKING ==========
def chunk_text(text, size=CHUNK_SIZE):
    return [text[i:i+size] for i in range(0, len(text), size)]

# ========== EMBEDDINGS ==========
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# ========== MAIN ==========
if __name__ == "__main__":
    print("📄 Extracting text from resume...")
    full_text = extract_text_from_pdf(RESUME_PATH)
    chunks = chunk_text(full_text)

    print(f"🔍 Generating embeddings for {len(chunks)} chunks...")
    embedded_chunks = []

    for chunk in tqdm(chunks):
        embedding = get_embedding(chunk)
        embedded_chunks.append({
            "text": chunk,
            "embedding": embedding
        })

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(embedded_chunks, f)

    print("✅ Done! Saved to resume_chunks.pkl")

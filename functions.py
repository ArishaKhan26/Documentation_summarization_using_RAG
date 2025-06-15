# functions.py

import spacy
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------- Setup -----------
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")  # Downloaded once, reused for all chunks

# ----------- Chunking -----------
def chunk_article(article, chunk_size=5, overlap=2):
    doc = nlp(article)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    i = 0
    while i + chunk_size <= len(sentences):
        chunk = sentences[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    if i < len(sentences):
        chunk = sentences[i:]
        chunks.append(" ".join(chunk))
    return chunks

def chunk_articles(articles, chunk_size=5, overlap=2):
    all_chunks = []
    for article in articles:
        chunks = chunk_article(article, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks

# ----------- Embedding using SentenceTransformers -----------
def get_embedding(text):
    return model.encode(text)

def embed_chunks(chunks): 
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return np.array(embeddings).astype("float32")

# ----------- FAISS Indexing -----------
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# ----------- Semantic Retrieval -----------
def retrieve_top_k_chunks(query, k, index, chunks):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

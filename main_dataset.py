# main.py

import os
import pandas as pd
import kagglehub
import time
from functions import chunk_articles, embed_chunks, build_faiss_index, retrieve_top_k_chunks
from transformers import pipeline, AutoTokenizer
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ------------------------------------------------------ PART A: DOCUMENT INGESTION --------------------------------------------------------

# ----------- Step 1: Download + Locate Dataset File -----------
print("\n Downloading ArXiv dataset from KaggleHub...")
path = kagglehub.dataset_download("Cornell-University/arxiv")

print("\n Contents of downloaded dataset folder:")
print(os.listdir(path))                        # see what files are there

# Expect a JSON file (not CSV) in this dataset
json_files = [f for f in os.listdir(path) if f.endswith(".json")]
if not json_files:
    raise FileNotFoundError(" No JSON file found in the dataset folder.")
dataset_file = os.path.join(path, json_files[0])
print(f" Using file: {dataset_file}")

# ----------- Step 2: Load a Sample of the Large JSON Dataset -----------
print("\n Loading dataset (streamed JSON lines)...")

sample_lines = []
with open(dataset_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 500:  # You can adjust this number to fit available memory
            break
        sample_lines.append(pd.read_json(line, lines=True))

df = pd.concat(sample_lines, ignore_index=True)

# Quick check
print(" Loaded records:", len(df))
print(" Columns:", df.columns.tolist())
print(df.head(2), "\n")

# Ensure title and abstract exist
df = df.dropna(subset=["title", "abstract"])

# Combine title and abstract for summarization
sample_articles = (df["title"] + ". " + df["abstract"]).tolist()[:10]



#----------------------------------------------------------- PART B: EMBEDDING RETRIEVAL-------------------------------------------------------

# ----------- Step 3: Chunk Articles -----------
print("\n Chunking articles...")
all_chunks = chunk_articles(sample_articles, chunk_size=5, overlap=2)

# ----------- Step 4: Embedding -----------
print("\n Generating embeddings...")
chunk_embeddings = embed_chunks(all_chunks)

# ----------- Step 5: Build FAISS Index -----------
print("\n Creating FAISS index...")
faiss_index = build_faiss_index(chunk_embeddings)

# ----------- Step 6: Semantic Retrieval Example -----------
query = "Summarize this document"
top_chunks = retrieve_top_k_chunks(query, k=5, index=faiss_index, chunks=all_chunks)

# ----------- Step 7: Display Top Retrieved Chunks -----------
print("\n Retrieved Chunks:")
for i, chunk in enumerate(top_chunks, 1):
    print(f"\nChunk {i}:\n{chunk[:300]}\n{'-'*60}")


#----------------------------------------------------------- PART C: SUMMARY GENERATION-------------------------------------------------------


# ----------- Step 8: Generate Final Summary -----------

print("\n Generating final summary from top retrieved chunks...")

# Combine top-k retrieved chunks into one document
combined_text = "\n".join(top_chunks)


# Load summarizer + tokenizer
model_name = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# ----------- Step 9: Token usage, Latency, and Similarity Scores. ---------

# Measure latency
start_time = time.time()
result = summarizer(combined_text, max_length=150, min_length=40, do_sample=False)[0]
end_time = time.time()

summary = result["summary_text"]

# Token usage
input_tokens = tokenizer(combined_text, return_tensors="pt")["input_ids"].shape[-1]
output_tokens = tokenizer(summary, return_tensors="pt")["input_ids"].shape[-1]

# Cosine similarity (average of retrieved vs query embedding)
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Small + fast

query_embedding = embedder.encode([query])
chunk_embeddings = embedder.encode(top_chunks)

similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
avg_similarity = np.mean(similarities)

# Output everything
print("\n Final Summary:\n")
print(summary)

print("\n Diagnostics:")
print(f" Latency: {end_time - start_time:.2f} seconds")
print(f" Tokens used — Input: {input_tokens}, Output: {output_tokens}")
print(f" Average cosine similarity (query vs top chunks): {avg_similarity:.4f}")

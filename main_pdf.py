# pdf_summary.py

import os
import time
import fitz  # PyMuPDF
from functions import chunk_articles, embed_chunks, build_faiss_index, retrieve_top_k_chunks
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------ PART A: PDF INGESTION --------------------------------------------------------

# ----------- Step 1: Load and Extract Text from PDF -----------
pdf_path = "sample.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(" PDF file not found. Please make sure 'sample.pdf' is in the project folder.")

print(f"\n Extracting text from PDF: {pdf_path}...")
pdf_text = ""
with fitz.open(pdf_path) as doc:
    for page in doc:
        pdf_text += page.get_text()

if not pdf_text.strip():
    raise ValueError(" Extracted text from PDF is empty. Check if the file is valid.")

sample_articles = [pdf_text]


# ------------------------------------------------------ PART B: EMBEDDING RETRIEVAL --------------------------------------------------------

# ----------- Step 2: Chunk Articles -----------
print("\n Chunking PDF text...")
all_chunks = chunk_articles(sample_articles, chunk_size=5, overlap=2)

# ----------- Step 3: Embedding -----------
print("\n Generating embeddings...")
chunk_embeddings = embed_chunks(all_chunks)

# ----------- Step 4: Build FAISS Index -----------
print("\n Creating FAISS index...")
faiss_index = build_faiss_index(chunk_embeddings)

# ----------- Step 5: Semantic Retrieval -----------
query = "Summarize this document"
top_chunks = retrieve_top_k_chunks(query, k=5, index=faiss_index, chunks=all_chunks)

# ----------- Step 6: Display Top Retrieved Chunks -----------
print("\n Retrieved Chunks:")
for i, chunk in enumerate(top_chunks, 1):
    print(f"\nChunk {i}:\n{chunk[:300]}\n{'-'*60}")


# ------------------------------------------------------ PART C: SUMMARY GENERATION --------------------------------------------------------

print("\n Generating final summary from top retrieved chunks...")
combined_text = "\n".join(top_chunks)

model_name = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

start_time = time.time()

# Truncate input to fit model's max token limit
max_input_tokens = 1024
tokenized_input = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=max_input_tokens)
truncated_text = tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)
result = summarizer(truncated_text, max_length=150, min_length=40, do_sample=False)[0]

end_time = time.time()
summary = result["summary_text"]

# ----------- Metrics -----------
input_tokens = tokenizer(combined_text, return_tensors="pt")["input_ids"].shape[-1]
output_tokens = tokenizer(summary, return_tensors="pt")["input_ids"].shape[-1]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
query_embedding = embedder.encode([query])
chunk_embeds = embedder.encode(top_chunks)
similarities = cosine_similarity(query_embedding, chunk_embeds)[0]
avg_similarity = np.mean(similarities)

# Output everything
print("\n Final Summary:\n")
print(summary)

print("\n Diagnostics:")
print(f" Latency: {end_time - start_time:.2f} seconds")
print(f" Tokens used — Input: {input_tokens}, Output: {output_tokens}")
print(f" Average cosine similarity (query vs top chunks): {avg_similarity:.4f}")

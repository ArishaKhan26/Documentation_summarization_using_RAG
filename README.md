# READ ME
### PDF Document Summarizer using Semantic Retrieval + LLM

This project implements a Retrieval-Augmented Generation (RAG) pipeline that takes a PDF file as input, extracts and chunks the content, embeds it semantically, retrieves the most relevant sections using a query, and generates a concise summary using a pre-trained language model (`facebook/bart-large-cnn`).

### Features
- Extracts text from any PDF file using PyMuPDF.
- Splits text into overlapping chunks for better semantic granularity.
- Embeds chunks using Sentence Transformers.
- Stores embeddings in a FAISS index for efficient semantic retrieval.
- Retrieves top `k` chunks relevant to a user query.
- Generates a summary using a pre-trained transformer model.
- Displays performance metrics including token usage and cosine similarity.

### Dataset
1)https://www.kaggle.com/datasets/Cornell-University/arxiv  OR
2)Any pdf of your own

### Project Structure
├── main_pdf.py # Main for pdf
├── main_dataset # Main for Kaggle dataset
├── functions.py # Contains helper functions: chunking, embedding, retrieval
├── sample.pdf # Your input PDF file 1
├── sample2.pdf # Your input PDF file 2
├── requirements.txt # List of required Python packages
├── description.docx #Details of each step
├── README.md # This file

### How to Run
Step 1: Set Up Your Project Folder
Step 2: Set Up Your Python Environment. Install requirements.txt
pip install -r requirements.txt
Step 3: Run the Main Script
python main_pdf.py


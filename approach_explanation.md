# Approach Explanation: Intelligent Document Analyst

Our system, the **Intelligent Document Analyst**, is designed as a modular, two-stage pipeline to extract and prioritize relevant information from documents based on a user's specific persona and task. This approach ensures high accuracy while adhering to strict performance and resource constraints.

### 1. Contextual Retrieval Stage 🎯

The first stage focuses on efficiently finding the most relevant content across all provided documents.

* **Query Formulation:** We first create a rich, contextual query by combining the user's `Persona` and `Job-to-be-Done`. This provides a deep semantic understanding of the user's intent.
* **Document Chunking:** Documents are parsed using `PyMuPDF` and intelligently broken down into semantic chunks (paragraphs). This preserves the context within each chunk.
* **Semantic Ranking:** We use the lightweight sentence-transformer model, **`all-MiniLM-L6-v2`** (~90MB), to convert both the query and all document chunks into vector embeddings. By calculating the cosine similarity between the query vector and each chunk vector, we can accurately score and rank every piece of text based on its relevance to the user's goal. This model is chosen for its excellent balance of speed, size, and performance on CPU.

### 2. Analysis & Refinement Stage 🔬

The second stage takes the top-ranked sections from the retrieval stage and performs a deeper analysis to generate concise, actionable insights.

* **Summarization:** For the highest-ranked chunks, we employ a distilled summarization model, **`sshleifer/distilbart-cnn-12-6`** (~480MB). This model creates brief, abstractive summaries of the key information contained within the most relevant text snippets.
* **Structured Output:** The final output is compiled into a structured JSON file, containing metadata, a ranked list of the most important sections (with document name, page, and rank), and the refined summaries for the top findings.

### Adherence to Constraints ✅

This entire system is designed for efficiency and offline execution. The total model size is **~570 MB**, well under the 1GB limit. Both models are optimized for fast CPU execution, ensuring the entire pipeline completes within the 60-second time constraint for a typical set of documents. The `Dockerfile` bakes all models into the container image during the build phase, guaranteeing that **no internet access is required at runtime**.
# document_analyst.py
import os
import fitz  # PyMuPDF
import json
import time
import re
from sentence_transformers import SentenceTransformer, util
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

class DocumentAnalyst:
    """An intelligent document analyst system to extract and prioritize sections."""
    def __init__(self, models_path='./models'):
        print("Initializing Document Analyst...")
        self.device = "cpu"  # Enforce CPU usage as per constraints

        # Load Sentence Transformer model for relevance ranking
        sbert_model_name = 'all-MiniLM-L6-v2'
        sbert_path = os.path.join(models_path, sbert_model_name)
        print(f"Loading Sentence Transformer model from {sbert_path}...")
        self.sbert_model = SentenceTransformer(sbert_path, device=self.device)

        # Load Summarization model for refinement
        summarizer_model_name = 'sshleifer/distilbart-cnn-12-6'
        summarizer_path = os.path.join(models_path, summarizer_model_name)
        print(f"Loading Summarization model from {summarizer_path}...")
        self.summarizer_tokenizer = BartTokenizer.from_pretrained(summarizer_path)
        self.summarizer_model = BartForConditionalGeneration.from_pretrained(summarizer_path).to(self.device)
        print("Initialization complete.")

    def _parse_and_chunk_pdfs(self, doc_paths):
        all_chunks = []
        for doc_path in doc_paths:
            doc_name = os.path.basename(doc_path)
            print(f"Processing document: {doc_name}")
            try:
                document = fitz.open(doc_path)
                for page_num, page in enumerate(document, 1):
                    text = page.get_text("text")
                    paragraphs = re.split(r'\n\s*\n', text)
                    for para in paragraphs:
                        clean_para = para.strip().replace('\n', ' ')
                        if len(clean_para) > 150: # Filter small text snippets
                            all_chunks.append({
                                'text': clean_para,
                                'doc_name': doc_name,
                                'page_num': page_num,
                                'section_title': f"Page {page_num} Snippet"
                            })
            except Exception as e:
                print(f"Could not process {doc_name}: {e}")
        return all_chunks

    def _create_query(self, persona, job_to_be_done):
        return f"As a {persona.get('role', '')} with expertise in {persona.get('expertise', '')}, I need to {job_to_be_done}."

    def _rank_chunks(self, query, chunks):
        print("Ranking chunks by relevance...")
        query_embedding = self.sbert_model.encode(query, convert_to_tensor=True, device=self.device)
        chunk_texts = [chunk['text'] for chunk in chunks]
        chunk_embeddings = self.sbert_model.encode(chunk_texts, convert_to_tensor=True, device=self.device)
        cosine_scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]

        for i, chunk in enumerate(chunks):
            chunk['score'] = cosine_scores[i].item()
        return sorted(chunks, key=lambda x: x['score'], reverse=True)

    def _summarize_text(self, text, max_length=150, min_length=30):
        inputs = self.summarizer_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        summary_ids = self.summarizer_model.generate(inputs, max_length=max_length, min_length=min_length, num_beams=4, early_stopping=True)
        return self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def _format_output(self, config, ranked_chunks, refined_data, timestamp):
        output = {
            "metadata": {
                "input_documents": config['document_paths'],
                "persona": config['persona'],
                "job_to_be_done": config['job_to_be_done'],
                "processing_timestamp": timestamp
            },
            "extracted_sections": [
                {
                    "document": chunk['doc_name'],
                    "page_number": chunk['page_num'],
                    "section_title": chunk['section_title'],
                    "importance_rank": i + 1,
                } for i, chunk in enumerate(ranked_chunks[:10]) # Top 10 extracted sections
            ],
            "sub_section_analysis": refined_data
        }
        return output

    def analyze(self, input_config_path):
        start_time = time.time()
        with open(input_config_path, 'r') as f:
            config = json.load(f)
        
        doc_paths = [os.path.join(os.path.dirname(input_config_path), p) for p in config['document_paths']]
        
        all_chunks = self._parse_and_chunk_pdfs(doc_paths)
        if not all_chunks:
            print("Error: No text could be extracted from the documents.")
            return None

        query = self._create_query(config['persona'], config['job_to_be_done'])
        print(f"\nGenerated Query: {query}\n")

        ranked_chunks = self._rank_chunks(query, all_chunks)
        
        print("\nGenerating refined text for top sections...")
        top_chunks_for_summary = ranked_chunks[:5] # Analyze the top 5
        refined_data = []
        for chunk in top_chunks_for_summary:
            summary = self._summarize_text(chunk['text'])
            refined_data.append({
                "document": chunk['doc_name'],
                "page_number": chunk['page_num'],
                "refined_text": summary
            })
        
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        final_output = self._format_output(config, ranked_chunks, refined_data, timestamp)
        
        print(f"\nAnalysis complete. Total processing time: {time.time() - start_time:.2f} seconds.")
        return final_output
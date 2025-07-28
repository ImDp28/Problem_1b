# download_models.py
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
import os

def download_and_save_models():
    """
    Downloads and saves the required models from Hugging Face to a local directory.
    This script is intended to be run once during the Docker image build process.
    """
    sbert_model_name = 'all-MiniLM-L6-v2'
    summarizer_model_name = 'sshleifer/distilbart-cnn-12-6'
    save_directory = './models'

    print(f"Creating directory: {save_directory}")
    os.makedirs(save_directory, exist_ok=True)

    # --- Download and save Sentence Transformer model ---
    sbert_path = os.path.join(save_directory, sbert_model_name)
    print(f"Downloading Sentence Transformer model: {sbert_model_name}...")
    sbert_model = SentenceTransformer(sbert_model_name)
    sbert_model.save(sbert_path)
    print(f"Sentence Transformer model saved to {sbert_path}")

    # --- Download and save Summarization model and tokenizer ---
    summarizer_path = os.path.join(save_directory, summarizer_model_name)
    print(f"Downloading Summarization model: {summarizer_model_name}...")
    tokenizer = BartTokenizer.from_pretrained(summarizer_model_name)
    model = BartForConditionalGeneration.from_pretrained(summarizer_model_name)
    tokenizer.save_pretrained(summarizer_path)
    model.save_pretrained(summarizer_path)
    print(f"Summarization model and tokenizer saved to {summarizer_path}")

    print("\n✅ All models downloaded and saved successfully.")

if __name__ == "__main__":
    download_and_save_models()
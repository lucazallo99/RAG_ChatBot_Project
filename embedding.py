import pandas as pd
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

# Load the parquet file containing the European history data
data_path = "C:/Users/lucaz/Downloads/BIA6304_Final/Zallo.Midterm.parquet"
df = pd.read_parquet(data_path)

# View the dataset
# Define the HuggingFaceEmbedding class
class HuggingFaceEmbedding:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
        self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, text: str) -> np.ndarray:
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings: np.ndarray = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings[0]
    
# Initialize the embedding model
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings for the 'content' column
embeddings = np.array([embedding_model(text) for text in df['content']])

# Set up FAISS index for fast similarity search
embedding_dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)  # Use L2 (Euclidean) distance
faiss_index.add(embeddings)  # Add embeddings to the FAISS index
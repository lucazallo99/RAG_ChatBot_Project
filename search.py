import pandas as pd
import numpy as np
from embedding import embedding_model, faiss_index, df

def search(query: str, top_k: int = 3) -> pd.DataFrame:

    # Generate embedding for the query
    query_embedding = embedding_model(query)

    # Perform the search in the FAISS index
    D, I = faiss_index.search(np.array([query_embedding]), k=top_k)

    # Create a DataFrame for the search results
    results = pd.DataFrame({
        "id": I[0],
        "title": df.iloc[I[0]]['title'].values,
        "content": df.iloc[I[0]]['content'].values,
        "similarity_score": D[0]  # Include the similarity scores if needed
    })

    return results
import os
import voyageai
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cosine
import numpy as np

# Set your Voyage API key
VOYAGE_API_KEY = "XXXXXXXX"  # Replace with your actual API Key


def get_embeddings_voyage_ai(texts, model="voyage-large-2-instruct"):
    """
    Fetch embeddings from the Voyage AI API for a list of texts.
    """
    token = os.environ.get("VOYAGE_API_KEY", VOYAGE_API_KEY)
    vo = voyageai.Client(api_key=token)

    try:
        response = vo.embed(texts, model=model, input_type=None)
        return [embedding for embedding in response.embeddings]
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        return []


def generate_embeddings_in_batches_voyage_ai(df, output_path, batch_size=100, model="voyage-embedding-large"):
    """
    Generate embeddings in batches for a DataFrame using Voyage AI and save to a pickle file.
    """
    # Check if embeddings file already exists
    if Path(output_path).exists():
        print(f"Embeddings file already exists at {output_path}. Loading embeddings...")
        return pd.read_pickle(output_path)

    # Generate embeddings in batches
    batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    all_embeddings = []
    for batch in batches:
        texts = batch['product_short_desc'].tolist()
        embeddings = get_embeddings_voyage_ai(texts, model=model)
        all_embeddings.extend(embeddings)
        print(f"Processed a batch of {len(texts)} texts.")

    # Add embeddings to DataFrame and save
    df['embedding'] = all_embeddings
    df.to_pickle(output_path)
    print(f"All embeddings generated and stored at {output_path}.")
    return df


def calculate_voyage_similarities(test_df, train_df, top_n=100):
    """
    Calculate cosine similarities for embeddings in test_df against train_df.
    Saves the top N similarities for each test item.
    """
    similarities = []

    # Create a progress bar for the outer loop
    outer_loop = tqdm(test_df.iloc[:top_n].iterrows(), total=top_n, desc="Calculating Voyage Similarities")

    for _, test_row in outer_loop:
        test_embedding = test_row['embedding']
        similarity = train_df.apply(
            lambda train_row: 1 - cosine(test_embedding, train_row['embedding']), axis=1
        )
        similarities.append(similarity)

    return similarities


def save_similarities(similarities, file_path):
    """
    Save the similarities list to a pickle file.
    """
    with open(file_path, 'wb') as f:
        pd.to_pickle(similarities, f)
    print(f"Similarities saved to {file_path}")

import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm
import pandas as pd

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return 1 - cosine(vec1, vec2)

def calculate_similarities(test_df, train_df, top_n=100):
    """
    Calculate cosine similarities for embeddings in test_df against train_df.
    Saves the top N similarities for each test item.
    """
    similarities = []

    # Create a progress bar for the outer loop
    outer_loop = tqdm(test_df.iloc[:top_n].iterrows(), total=top_n, desc="Calculating Similarities")

    for _, test_row in outer_loop:
        test_embedding = test_row['embedding']
        similarity = train_df.apply(
            lambda train_row: cosine_similarity(test_embedding, train_row['embedding']), axis=1
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

def load_similarities(file_path):
    """
    Load the similarities list from a pickle file.
    """
    return pd.read_pickle(file_path)

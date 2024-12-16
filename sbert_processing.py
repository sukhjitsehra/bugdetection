import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

# SBERT model name
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # Set the model to evaluation mode


def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling to get sentence embeddings.
    """
    token_embeddings = model_output['last_hidden_state']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_embeddings_sbert(texts):
    """
    Generate embeddings using SBERT.
    """
    # Tokenize and prepare input
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}
    
    # Compute embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return embeddings.cpu().numpy()


def generate_embeddings_in_batches_sbert(df, output_path, batch_size=25):
    """
    Generate SBERT embeddings in batches and save to a pickle file.
    """
    # Check if embeddings file already exists
    if Path(output_path).exists():
        print(f"Embeddings file already exists at {output_path}. Loading embeddings...")
        return pd.read_pickle(output_path)

    all_embeddings = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        texts = batch['product_short_desc'].tolist()
        embeddings = get_embeddings_sbert(texts)
        all_embeddings.extend(embeddings)
        print(f"Processed a batch of {len(texts)} texts.")

    # Add embeddings to DataFrame and save
    df['embedding'] = list(all_embeddings)
    df.to_pickle(output_path)
    print(f"All embeddings generated and stored at {output_path}.")
    return df


def calculate_sbert_similarities(test_df, train_df, top_n=100):
    """
    Calculate cosine similarities for SBERT embeddings.
    """
    similarities = []

    # Progress bar for processing
    outer_loop = tqdm(test_df.iloc[:top_n].iterrows(), total=top_n, desc="Calculating SBERT Similarities")

    for _, test_row in outer_loop:
        test_embedding = test_row['embedding']
        similarity = train_df.apply(
            lambda train_row: 1 - cosine(test_embedding, train_row['embedding']), axis=1
        )
        similarities.append(similarity)

    return similarities

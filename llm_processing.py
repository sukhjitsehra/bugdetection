import requests
import pandas as pd
from pathlib import Path


API_KEY = "XXXXXXXX"  # Replace with your API Key
API_ENDPOINT = 'https://api.openai.com/v1/embeddings'

def get_embeddings(texts, model="text-embedding-3-large"):
    """
    Fetch embeddings from the OpenAI API for a list of texts.
    """
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    data = {
        "input": texts,
        "model": model,
    }

    response = requests.post(API_ENDPOINT, headers=headers, json=data)

    if response.status_code == 200:
        return [item['embedding'] for item in response.json()['data']]
    else:
        print(f"Error during API call: {response.status_code}")
        print(response.text)
        return []

def generate_embeddings_in_batches(df, output_path, batch_size=200, model="text-embedding-3-large"):
    """
    Generate embeddings in batches for a DataFrame and save to a pickle file.
    """
    # Check if embeddings file already exists
    if Path(output_path).exists():
        print(f"Embeddings file already exists at {output_path}. Loading embeddings...")
        return pd.read_pickle(output_path)

    # Generate embeddings in batches
    batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
    all_embeddings = []
    for batch in batches:
        texts = batch['product_short_desc'].tolist()
        embeddings = get_embeddings(texts, model=model)
        all_embeddings.extend(embeddings)
        print(f"Processed a batch of {len(texts)} texts.")

    # Add embeddings to DataFrame and save
    df['embedding'] = all_embeddings
    df.to_pickle(output_path)
    print(f"All embeddings generated and stored at {output_path}.")
    return df

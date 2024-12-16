import numpy as np
import pandas as pd

def combine_similarities(openai_similarities, voyage_similarities):
    """
    Combine similarities from OpenAI and Voyage AI models.
    """
    combined_similarities = [
        [x + y for x, y in zip(voyage_row, openai_row)]
        for voyage_row, openai_row in zip(voyage_similarities, openai_similarities)
    ]
    return combined_similarities


def generate_matrices(combined_similarities, train_data, size_limit=100):
    """
    Generate issue ID and similarity score matrices based on combined similarities.
    """
    issue_id_matrix = []
    score_matrix = []

    for i, row in enumerate(combined_similarities):
        # Get the indices of the top `size_limit` elements
        top_indices = np.argsort(row)[-size_limit:]
        
        # Retrieve IDs and scores
        issue_id_matrix.append([train_data["id"].iloc[j] for j in top_indices])
        score_matrix.append(np.sort(row)[-size_limit:])
    
    # Reverse each inner list for descending order
    issue_id_matrix = [ids[::-1] for ids in issue_id_matrix]
    score_matrix = [scores[::-1] for scores in score_matrix]

    return issue_id_matrix, score_matrix


def save_matrices(issue_id_matrix, score_matrix, issue_id_file, score_file):
    """
    Save the matrices to text files.
    """
    np.savetxt(issue_id_file, issue_id_matrix, fmt='%s')
    np.savetxt(score_file, score_matrix, fmt='%s')
    print(f"Matrices saved to {issue_id_file} and {score_file}")

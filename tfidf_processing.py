from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def generate_tfidf_vectors(corpus_train, corpus_test, ngram_range=(2, 2), min_df=2):
    """
    Generate TF-IDF vectors for train and test corpora.
    """
    tfidf_transformer = TfidfVectorizer(stop_words='english', min_df=min_df, ngram_range=ngram_range)
    tfidf_train = tfidf_transformer.fit_transform(corpus_train)
    tfidf_test = tfidf_transformer.transform(corpus_test)
    return tfidf_train, tfidf_test


def calculate_tfidf_similarities(tfidf_train, tfidf_test, test_data):
    """
    Calculate cosine similarities between TF-IDF vectors for train and test datasets.
    """
    tfidf_matrix_test = [[] for _ in range(len(test_data))]
    for i in range(len(test_data)):
        temp = cosine_similarity(tfidf_test[i], tfidf_train)[0]
        tfidf_matrix_test[i] = temp.tolist()
    return tfidf_matrix_test


def combine_tfidf_and_llm(tfidf_similarities, llm_similarities):
    """
    Combine TF-IDF similarities with LLM similarities.
    """
    combined_similarities = []
    for tfidf_row, llm_row in zip(tfidf_similarities, llm_similarities):
        combined_row = [x + y for x, y in zip(tfidf_row, llm_row)]
        combined_similarities.append(combined_row)
    return combined_similarities


def generate_similarity_matrices(combined_similarities, train_data, size_limit=100):
    """
    Generate issue ID and similarity score matrices from combined similarities.
    """
    issueid_matrix = []
    score_matrix = []

    for combined_row in combined_similarities:
        # Get indices of top similarities
        top_indices = np.argsort(combined_row)[-size_limit:]
        # Retrieve issue IDs and scores
        issueid_matrix.append([train_data["id"][j] for j in top_indices][::-1])
        score_matrix.append(np.sort(combined_row)[-size_limit:][::-1])

    return issueid_matrix, score_matrix

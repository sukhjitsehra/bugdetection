from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm
import numpy as np
import pandas as pd


def generate_glove_vectors(train_data, test_data, glove_model_path="GoogleNews-vectors-negative300.bin.gz", size_limit=300):
    """
    Generates GloVe vectors for train and test datasets.
    """
    # Load the GloVe model
    model_glove = KeyedVectors.load_word2vec_format(glove_model_path, binary=True)

    # Generate GloVe vectors for train data
    V_gl_train = csr_matrix(sent_vectorizer(train_data['Key_words'].iloc[0], model_glove, size_limit).reshape(-1, 1))
    for i in tqdm(range(1, len(train_data)), desc="Generating GloVe vectors for train data"):
        vector = csr_matrix(sent_vectorizer(train_data['Key_words'].iloc[i], model_glove, size_limit).reshape(-1, 1))
        V_gl_train = hstack((V_gl_train, vector))
    V_gl_train = csr_matrix(V_gl_train.T)

    # Generate GloVe vectors for test data
    V_gl_test = csr_matrix(sent_vectorizer(test_data['Key_words'].iloc[0], model_glove, size_limit).reshape(-1, 1))
    for i in tqdm(range(1, len(test_data)), desc="Generating GloVe vectors for test data"):
        vector = csr_matrix(sent_vectorizer(test_data['Key_words'].iloc[i], model_glove, size_limit).reshape(-1, 1))
        V_gl_test = hstack((V_gl_test, vector))
    V_gl_test = csr_matrix(V_gl_test.T)

    return V_gl_train, V_gl_test


def sent_vectorizer(sent, model, size_limit):
    """
    Converts a sentence into a vector by averaging GloVe embeddings of its words.
    """
    sent_vec = np.zeros(size_limit)
    for word in sent:
        try:
            word_vec = model[word][:size_limit]
            sent_vec = np.add(sent_vec, word_vec)
        except KeyError:
            pass
    return np.nan_to_num(sent_vec / np.linalg.norm(sent_vec), nan=0)


def calculate_glove_similarities(glove_train, glove_test, test_data):
    """
    Calculate cosine similarities between GloVe vectors for train and test datasets.
    """
    glove_matrix_test = [[] for _ in range(len(test_data))]
    for i in tqdm(range(len(test_data)), desc="Calculating GloVe similarities"):
        similarities = cosine_similarity(glove_test[i], glove_train)[0]
        glove_matrix_test[i] = similarities.tolist()
    return glove_matrix_test


def combine_glove_and_llm(glove_similarities, llm_similarities):
    """
    Combine GloVe similarities with LLM similarities.
    """
    combined_similarities = []
    for glove_row, llm_row in zip(glove_similarities, llm_similarities):
        combined_row = [x + y for x, y in zip(glove_row, llm_row)]
        combined_similarities.append(combined_row)
    return combined_similarities


def save_glove_results(similarities, train_data, matrix_file, issue_id_file, size_limit=100):
    """
    Save similarity results to files for GloVe-only and combined similarities.
    """
    issue_id_matrix = []
    score_matrix = []

    for similarity_row in similarities:
        top_indices = np.argsort(similarity_row)[-size_limit:]
        issue_id_matrix.append([train_data["id"][j] for j in top_indices][::-1])
        score_matrix.append(np.sort(similarity_row)[-size_limit:][::-1])

    np.savetxt(matrix_file, score_matrix, fmt='%s')
    np.savetxt(issue_id_file, issue_id_matrix, fmt='%s')

    print(f"Saved similarity scores to {matrix_file}")
    print(f"Saved issue IDs to {issue_id_file}")

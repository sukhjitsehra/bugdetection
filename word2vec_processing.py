import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec, Phrases
from gensim.models.phrases import Phraser
from tqdm import tqdm
import pandas as pd


def sent_vectorizer(sent, model, size_limit):
    """
    Converts a sentence into a vector by averaging Word2Vec embeddings of its words.
    """
    sent_vec = np.zeros(size_limit)
    for word in sent:
        try:
            word_vec = model.wv[word][:size_limit]
            sent_vec = np.add(sent_vec, word_vec)
        except KeyError:
            pass
    return np.nan_to_num(sent_vec / np.linalg.norm(sent_vec), nan=0)


def generate_word2vec_vectors(train_data, test_data, size_limit=100):
    """
    Generates Word2Vec vectors for train and test datasets.
    """
    # Prepare sentences for Word2Vec training
    train_sentences = train_data['Key_words'].tolist()
    test_sentences = test_data['Key_words'].tolist()

    # Create bigrams
    bigram = Phrases(train_sentences + test_sentences, min_count=1, threshold=2)
    bigram_phraser = Phraser(bigram)

    for sentences in [train_sentences, test_sentences]:
        for i, sent in enumerate(sentences):
            sentences[i].extend(bigram_phraser[sent])
            sentences[i] = list(set(sentences[i]))

    # Train Word2Vec model
    model = Word2Vec(train_sentences, vector_size=size_limit, sg=1, min_count=1)

    # Generate Word2Vec vectors for train data
    V_wv_train = []
    for sentence in tqdm(train_sentences, desc="Generating Word2Vec vectors for train data"):
        V_wv_train.append(sent_vectorizer(sentence, model, size_limit))
    V_wv_train = csr_matrix(V_wv_train)

    # Generate Word2Vec vectors for test data
    V_wv_test = []
    for sentence in tqdm(test_sentences, desc="Generating Word2Vec vectors for test data"):
        V_wv_test.append(sent_vectorizer(sentence, model, size_limit))
    V_wv_test = csr_matrix(V_wv_test)

    return V_wv_train, V_wv_test


def calculate_word2vec_similarities(V_wv_train, V_wv_test, test_data):
    """
    Calculate cosine similarities between Word2Vec vectors for train and test datasets.
    """
    w2v_matrix_test = [[] for _ in range(len(test_data))]
    for i in tqdm(range(len(test_data)), desc="Calculating Word2Vec similarities"):
        similarities = cosine_similarity(V_wv_test[i], V_wv_train)[0]
        w2v_matrix_test[i] = similarities.tolist()
    return w2v_matrix_test


def combine_word2vec_and_llm(w2v_similarities, llm_similarities):
    """
    Combine Word2Vec similarities with LLM similarities.
    """
    combined_similarities = []
    for w2v_row, llm_row in zip(w2v_similarities, llm_similarities):
        combined_row = [x + y for x, y in zip(w2v_row, llm_row)]
        combined_similarities.append(combined_row)
    return combined_similarities


def save_word2vec_results(similarities, train_data, matrix_file, issueid_file, size_limit=100):
    """
    Save similarity results to files for Word2Vec-only and combined similarities.
    """
    issueid_matrix = []
    score_matrix = []

    for similarity_row in similarities:
        top_indices = np.argsort(similarity_row)[-size_limit:]
        issueid_matrix.append([train_data["id"][j] for j in top_indices][::-1])
        score_matrix.append(np.sort(similarity_row)[-size_limit:][::-1])

    np.savetxt(matrix_file, score_matrix, fmt='%s')
    np.savetxt(issueid_file, issueid_matrix, fmt='%s')

    print(f"Saved similarity scores to {matrix_file}")
    print(f"Saved issue IDs to {issueid_file}")

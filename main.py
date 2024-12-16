import logging
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import setup_nltk_dependencies, prepare_datasets
from contractions import contractions_dict
from llm_processing import generate_embeddings_in_batches as generate_openai_embeddings
from similarity_calculation import calculate_similarities, save_similarities
from voyage_processing import generate_embeddings_in_batches_voyage_ai, calculate_voyage_similarities
from sbert_processing import generate_embeddings_in_batches_sbert, calculate_sbert_similarities
from tfidf_processing import (
    generate_tfidf_vectors,
    calculate_tfidf_similarities,
    combine_tfidf_and_llm,
    generate_similarity_matrices,
)
from word2vec_processing import (
    generate_word2vec_vectors,
    calculate_word2vec_similarities,
    combine_word2vec_and_llm,
    save_word2vec_results,
)
from glove_processing import (
    generate_glove_vectors,
    calculate_glove_similarities,
    combine_glove_and_llm,
    save_glove_results,
)
from combined_processing import combine_similarities, generate_matrices, save_matrices


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_recall(predictions_matrix, ground_truth_list, k_range=range(1, 51)):
    """
    Calculate the recall rate for a range of k values.
    """
    recall_rates = []

    for k in k_range:
        total_count = 0
        correct_predictions = 0

        for i in range(len(predictions_matrix)):
            top_k_predictions = predictions_matrix[i][:k]
            actual_dup_list = ground_truth_list[i]

            total_count += len(actual_dup_list)
            matching_predictions = [pred for pred in top_k_predictions if pred in actual_dup_list]
            correct_predictions += len(matching_predictions)
        
        recall_rate = correct_predictions / total_count
        recall_rates.append(recall_rate)

    return recall_rates


def calculate_map(predictions_matrix, ground_truth_list, k_range=range(1, 51)):
    """
    Calculate the Mean Average Precision (MAP) for a range of k values.
    """
    map_scores = []
    for k in k_range:
        average_precisions = []
        for i in range(len(predictions_matrix)):
            top_k_predictions = predictions_matrix[i][:k]
            actual_dup_list = ground_truth_list[i]
            num_relevant = 0
            precision_at_relevant = []
            for j, pred in enumerate(top_k_predictions):
                if pred in actual_dup_list:
                    num_relevant += 1
                    precision_at_relevant.append(num_relevant / (j + 1))
            if precision_at_relevant:
                average_precision = sum(precision_at_relevant) / len(actual_dup_list)
            else:
                average_precision = 0
            average_precisions.append(average_precision)
        map_score = sum(average_precisions) / len(average_precisions)
        map_scores.append(map_score)
    return map_scores


def calculate_mrr(predictions_matrix, ground_truth_list, k_range=range(1, 51)):
    """
    Calculate the Mean Reciprocal Rank (MRR) for a range of k values.
    """
    mrr_scores = []
    for k in k_range:
        reciprocal_ranks = []
        for i in range(len(predictions_matrix)):
            top_k_predictions = predictions_matrix[i][:k]
            actual_dup_list = ground_truth_list[i]
            rank_found = False
            for j, pred in enumerate(top_k_predictions):
                if pred in actual_dup_list:
                    reciprocal_ranks.append(1 / (j + 1))
                    rank_found = True
                    break
            if not rank_found:
                reciprocal_ranks.append(0)
        mrr_score = sum(reciprocal_ranks) / len(reciprocal_ranks)
        mrr_scores.append(mrr_score)
    return mrr_scores



if __name__ == "__main__":
    # Step 1: Setup dependencies and load dataset
    logging.info("Setting up NLTK dependencies...")
    setup_nltk_dependencies()

    dataset_path = "kde.csv"
    logging.info(f"Loading dataset from {dataset_path}...")
    train_data, test_data = prepare_datasets(dataset_path, contractions_dict)

    # Step 2: Generate embeddings and similarities for OpenAI
    logging.info("Generating OpenAI embeddings for training data...")
    openai_train_embeddings_path = "kdeembeddings_openaitrainingmodel.pkl"
    openai_test_embeddings_path = "kdeembeddings_openaitestingmodel.pkl"
    llmtraindf = generate_openai_embeddings(train_data, openai_train_embeddings_path)
    llmtestdf = generate_openai_embeddings(test_data, openai_test_embeddings_path)

    logging.info("Calculating OpenAI similarities...")
    openai_similarities = calculate_similarities(llmtestdf, llmtraindf, top_n=100)
    openai_similarities_file = "kdellm_similarities.pkl"
    save_similarities(openai_similarities, openai_similarities_file)

    # Step 3: Generate embeddings and similarities for Voyage AI
    logging.info("Generating Voyage AI embeddings for training data...")
    voyage_train_embeddings_path = "kdeembeddings_voyagetrainingmodel.pkl"
    voyage_test_embeddings_path = "kdeembeddings_voyagetestingmodel.pkl"
    voyagetraindf = generate_embeddings_in_batches_voyage_ai(train_data, voyage_train_embeddings_path)
    voyagetestdf = generate_embeddings_in_batches_voyage_ai(test_data, voyage_test_embeddings_path)

    logging.info("Calculating Voyage AI similarities...")
    voyage_similarities = calculate_voyage_similarities(voyagetestdf, voyagetraindf, top_n=100)
    voyage_similarities_file = "kdevoyagellm_similarities.pkl"
    save_similarities(voyage_similarities, voyage_similarities_file)

    # Step 4: Generate embeddings and similarities for SBERT
    logging.info("Generating SBERT embeddings for training data...")
    sbert_train_embeddings_path = "kdeembeddings_sberttraining.pkl"
    sbert_test_embeddings_path = "kdeembeddings_sberttesting.pkl"
    sberttraindf = generate_embeddings_in_batches_sbert(train_data, sbert_train_embeddings_path)
    sberttestdf = generate_embeddings_in_batches_sbert(test_data, sbert_test_embeddings_path)

    logging.info("Calculating SBERT similarities...")
    sbert_similarities = calculate_sbert_similarities(sberttestdf, sberttraindf, top_n=100)
    sbert_similarities_file = "kdesbert_similarities.pkl"
    save_similarities(sbert_similarities, sbert_similarities_file)

    # Step 5: TF-IDF Processing
    logging.info("Generating TF-IDF vectors for train and test datasets...")
    tfidf_train, tfidf_test = generate_tfidf_vectors(train_data['corpus'], test_data['corpus'])

    logging.info("Calculating TF-IDF similarities...")
    tfidf_similarities = calculate_tfidf_similarities(tfidf_train, tfidf_test, test_data)

    logging.info("Saving TF-IDF-only results...")
    tfidf_only_issueid_matrix, tfidf_only_score_matrix = generate_similarity_matrices(
        tfidf_similarities, train_data, size_limit=100
    )
    np.savetxt('kde_only_tfidf_matrix_test.txt', tfidf_only_score_matrix, fmt='%s')
    np.savetxt('kde_only_tfidf_issueid_matrix_test.txt', tfidf_only_issueid_matrix, fmt='%s')

    # Step 6: Word2Vec Processing
    logging.info("Generating Word2Vec vectors for train and test datasets...")
    w2v_train, w2v_test = generate_word2vec_vectors(train_data, test_data)

    logging.info("Calculating Word2Vec similarities...")
    w2v_similarities = calculate_word2vec_similarities(w2v_train, w2v_test, test_data)

    logging.info("Saving Word2Vec-only results...")
    save_word2vec_results(
        w2v_similarities, train_data,
        "kde_only_w2v_matrix_test.txt", "kde_only_w2v_issueid_matrix_test.txt"
    )

    # Step 7: GloVe Processing
    logging.info("Generating GloVe vectors for train and test datasets...")
    glove_train, glove_test = generate_glove_vectors(train_data, test_data)

    logging.info("Calculating GloVe similarities...")
    glove_similarities = calculate_glove_similarities(glove_train, glove_test, test_data)

    logging.info("Saving GloVe-only results...")
    save_glove_results(
        glove_similarities, train_data,
        "kde_only_glove_matrix_test.txt", "kde_only_glove_issue_id_matrix_test.txt"
    )

    # Step 8: Combine Similarities
    logging.info("Combining OpenAI and Voyage similarities...")
    combined_openai_voyage_similarities = combine_similarities(openai_similarities, voyage_similarities)
    save_matrices(*generate_matrices(combined_openai_voyage_similarities, train_data, size_limit=100),
                  "kde_openai_voyage_matrix_test.txt", "kde_openai_voyage_issueid_matrix_test.txt")

    logging.info("Combining Voyage and SBERT similarities...")
    combined_voyage_sbert_similarities = combine_similarities(voyage_similarities, sbert_similarities)
    save_matrices(*generate_matrices(combined_voyage_sbert_similarities, train_data, size_limit=100),
                  "kde_voyage_sbert_matrix_test.txt", "kde_voyage_sbert_issueid_matrix_test.txt")

    logging.info("Combining OpenAI and TF-IDF similarities...")
    combined_tfidf_llm_similarities = combine_tfidf_and_llm(tfidf_similarities, openai_similarities)
    save_matrices(*generate_matrices(combined_tfidf_llm_similarities, train_data, size_limit=100),
                  "kde_openai_tfidf_matrix_test.txt", "kde_openai_tfidf_issueid_matrix_test.txt")

    logging.info("Combining OpenAI and Word2Vec similarities...")
    combined_w2v_openai_similarities = combine_word2vec_and_llm(w2v_similarities, openai_similarities)
    save_word2vec_results(
        combined_w2v_openai_similarities, train_data,
        "kde_w2v_openai_matrix_test.txt", "kde_w2v_openai_issueid_matrix_test.txt"
    )

    logging.info("Combining OpenAI and GloVe similarities...")
    combined_glove_openai_similarities = combine_glove_and_llm(glove_similarities, openai_similarities)
    save_glove_results(
        combined_glove_openai_similarities, train_data,
        "kde_glove_openai_matrix_test.txt", "kde_glove_openai_issue_id_matrix_test.txt"
    )

    logging.info("Execution complete. All models processed and results saved.")

    # Metric logs
    logging.info("Loading prediction matrices for metrics calculation...")
    models_matrices = {
        "OpenAI + Voyage": (
            np.loadtxt('kde_openai_voyage_matrix_test.txt', dtype=float),
            np.loadtxt('kde_openai_voyage_issueid_matrix_test.txt', dtype=int),
        ),
        "Voyage + SBERT": (
            np.loadtxt('kde_sbert_voyage_matrix_test.txt', dtype=float),
            np.loadtxt('kde_sbert_voyage_issueid_matrix_test.txt', dtype=int),
        ),
        "TF-IDF + OpenAI": (
            np.loadtxt('kde_openai_tf_idf_matrix_test.txt', dtype=float),
            np.loadtxt('kde_openai_tfidf_issueid_matrix_test.txt', dtype=int),
        ),
        "Word2Vec + OpenAI": (
            np.loadtxt('kde_w2v_openai_matrix_test.txt', dtype=float),
            np.loadtxt('kde_w2v_openai_issueid_matrix_test.txt', dtype=int),
        ),
        "GloVe + OpenAI": (
            np.loadtxt('kde_glove_openai_matrix_test.txt', dtype=float),
            np.loadtxt('kde_glove_openai_issue_id_matrix_test.txt', dtype=int),
        ),
        "TF-IDF Only": (
            np.loadtxt('kde_only_tfidf_matrix_test.txt', dtype=float),
            np.loadtxt('kde_only_tfidf_issueid_matrix_test.txt', dtype=int),
        ),
        "Word2Vec Only": (
            np.loadtxt('kde_only_w2v_matrix_test.txt', dtype=float),
            np.loadtxt('kde_only_w2v_issueid_matrix_test.txt', dtype=int),
        ),
        "GloVe Only": (
            np.loadtxt('kde_only_glove_matrix_test.txt', dtype=float),
            np.loadtxt('kde_only_glove_issue_id_matrix_test.txt', dtype=int),
        ),
    }

    ground_truth = test_data['dup_list']

    # Calculate and plot metrics
    metrics = {"Recall": calculate_recall, "MAP": calculate_map, "MRR": calculate_mrr}
    for metric_name, metric_fn in metrics.items():
        logging.info(f"Calculating {metric_name} for all models...")
        plt.figure(figsize=(10, 6))
        for model_name, (matrix_test, _) in models_matrices.items():
            metric_scores = metric_fn(matrix_test, ground_truth)
            plt.plot(metric_scores, label=model_name)
        plt.title(f"{metric_name} for Different Models and Scenarios")
        plt.xlabel("k values")
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.savefig(f"kde_{metric_name.lower()}_comparison.png", bbox_inches="tight")
        plt.show()

    logging.info("All metrics calculated, plotted, and saved.")

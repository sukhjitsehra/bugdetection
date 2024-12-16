# **Bug Detection and Similarity Analysis Using LLMs and Traditional Models**

This project implements a **bug detection and similarity analysis pipeline** by leveraging various **Large Language Models (LLMs)** and **traditional NLP techniques**. Models such as OpenAI embeddings, Voyage AI, SBERT, TF-IDF, Word2Vec, and GloVe are utilized to identify potential duplicates and calculate metrics such as **Recall**, **Mean Average Precision (MAP)**, and **Mean Reciprocal Rank (MRR)**.

## **Project Overview**
The pipeline includes:
1. **Preprocessing Bug Report Data**:
   - Data cleaning, tokenization, lemmatization, and contraction expansion.
   - Generation of `train_data` and `test_data` for modeling.

2. **Model Implementation**:
   - Generating embeddings using:
     - OpenAI (`text-embedding-3-large` model).
     - Voyage AI (`voyage-large-2-instruct` model).
     - SBERT (`sentence-transformers/all-MiniLM-L6-v2`).
   - Traditional vector-based methods:
     - TF-IDF.
     - Word2Vec.
     - GloVe.

3. **Similarity Calculation**:
   - Cosine similarity between embeddings.
   - Combined similarity matrices for hybrid models like:
     - OpenAI + Voyage.
     - OpenAI + TF-IDF.
     - OpenAI + Word2Vec.
     - OpenAI + GloVe.
     - Voyage + SBERT.

4. **Evaluation Metrics**:
   - **Recall Rate**: Measures the proportion of true positives retrieved.
   - **MAP (Mean Average Precision)**: Evaluates ranking precision at all relevant positions.
   - **MRR (Mean Reciprocal Rank)**: Evaluates ranking efficiency.

5. **Visualization**:
   - Comparison of metrics across models using plots.

---

## **Project Structure**

```plaintext
.
├── data_preprocessing.py       # Data preprocessing logic
├── text_processing.py          # Text cleaning and processing functions
├── llm_processing.py           # OpenAI embedding generation and similarity calculation
├── voyage_processing.py        # Voyage AI embedding generation and similarity calculation
├── sbert_processing.py         # SBERT embedding generation and similarity calculation
├── tfidf_processing.py         # TF-IDF vector generation and integration with LLMs
├── word2vec_processing.py      # Word2Vec vector generation and integration with LLMs
├── glove_processing.py         # GloVe vector generation and integration with LLMs
├── combined_processing.py      # Combination logic for hybrid models
├── similarity_calculation.py   # Generic similarity computation and file handling
├── main.py                     # Main execution script
├── kde.csv                     # Dataset for bug reports
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/bug-detection-llm.git
cd bug-detection-llm
```

### **2. Install Dependencies**
Create a virtual environment and install the dependencies:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

pip install -r requirements.txt
```

### **3. Download Pre-trained Models**
Ensure you have the required models downloaded:
- GloVe embeddings (`GoogleNews-vectors-negative300.bin.gz`).
- Word2Vec and other dependencies will be handled by the scripts.

---

## **Execution Steps**

### **1. Preprocess the Data**
The dataset `kde.csv` is preprocessed to generate train and test splits:
```bash
python main.py
```

### **2. Model Execution**
The script will:
- Generate embeddings for each model.
- Compute similarity matrices.
- Save results as `.txt` files in the project directory.

### **3. Metrics and Plots**
After all models are executed, the script:
- Calculates **Recall**, **MAP**, and **MRR** metrics.
- Saves plots comparing metrics for all models as `.png` files.

---

## **Results**
The results include:
1. **Similarity Matrices**:
   - `kde_openai_voyage_matrix_test.txt`, `kde_only_tfidf_matrix_test.txt`, etc.

2. **Evaluation Plots**:
   - `kde_recall_rate_comparison.png`
   - `kde_map_comparison.png`
   - `kde_mrr_comparison.png`

3. **Metrics Comparison**:
   The metrics plots compare performance across all combinations:
   - **OpenAI + Voyage**
   - **Voyage + SBERT**
   - **TF-IDF + OpenAI**
   - **Word2Vec + OpenAI**
   - **GloVe + OpenAI**
   - **TF-IDF Only**
   - **Word2Vec Only**
   - **GloVe Only**

---

## **Key Functions**

| File                 | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `data_preprocessing` | Prepares `train_data` and `test_data` by processing text.                   |
| `llm_processing`     | Generates embeddings using OpenAI and calculates similarities.             |
| `voyage_processing`  | Generates embeddings using Voyage AI and calculates similarities.          |
| `sbert_processing`   | Generates embeddings using SBERT and calculates similarities.              |
| `tfidf_processing`   | Generates TF-IDF vectors and combines with OpenAI similarities.            |
| `word2vec_processing`| Generates Word2Vec embeddings and combines with OpenAI similarities.       |
| `glove_processing`   | Generates GloVe embeddings and combines with OpenAI similarities.          |
| `main.py`            | Runs the entire pipeline and saves results.                                |

---


## **Dependencies**
- Python 3.8+
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `gensim`
  - `nltk`
  - `tqdm`
  - `requests`
  - `transformers`

Install them using `pip install -r requirements.txt`.

---

## **Contributions**
Contributions are welcome! Feel free to open an issue or create a pull request.

---

## **License**
This project is licensed under the [MIT License](LICENSE).

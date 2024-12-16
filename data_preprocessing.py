import pandas as pd
import nltk
from text_processing import preprocess_text, remove_large_values
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup NLTK dependencies
def setup_nltk_dependencies():
    """
    Downloads required NLTK resources if not already present.
    """
    nltk_dependencies = [
        "stopwords", "wordnet", "brown", "names", 
        "averaged_perceptron_tagger", "universal_tagset"
    ]
    for dependency in nltk_dependencies:
        nltk.download(dependency)
    print("NLTK dependencies downloaded.")

def preprocess_corpus(data, column, contractions_dict):
    """
    Preprocess text column and return the processed corpus.
    """
    corpus = []
    for text in data[column]:
        processed_text = preprocess_text(text, contractions_dict)
        corpus.append(processed_text)
    return corpus

def prepare_datasets(data_path, contractions_dict):
    """
    Load and preprocess datasets, creating train and test sets.
    """
    # Load dataset
    data = pd.read_csv(data_path)

    # Create product short descriptions
    data['product_short_desc'] = data['product'] + ' - ' + data['component'] + ' - ' + data['short_desc']

    # Split data into train and test sets
    test_data = data[data['dup_list'].notnull()]
    train_data = data[data['dup_list'].isnull()]

    # Remove invalid short descriptions
    test_data = test_data[~test_data['product_short_desc'].apply(lambda x: isinstance(x, float))]
    train_data = train_data[~train_data['product_short_desc'].apply(lambda x: isinstance(x, float))]

    # Reset indices
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    train_data = train_data.head(50000)

    # Preprocess test_data
    test_data['dup_list'] = test_data['dup_list'].apply(remove_large_values)
    test_data = test_data[test_data['dup_list'].astype(bool)]
    
    # Preprocess corpus for train and test datasets
    train_data['corpus'] = preprocess_corpus(train_data, 'short_desc', contractions_dict)
    test_data['corpus'] = preprocess_corpus(test_data, 'short_desc', contractions_dict)

    # Generate key words for train and test datasets
    train_data['Key_words'] = train_data['corpus'].map(lambda x: x.lower().split(' '))
    test_data['Key_words'] = test_data['corpus'].map(lambda x: x.lower().split(' '))

    # Process product and component columns
    train_data['product'] = train_data['product'].map(lambda x: x.lower().split(' '))
    test_data['product'] = test_data['product'].map(lambda x: x.lower().split(' '))

    train_data['component'] = train_data['component'].map(lambda x: x.lower().split(' '))
    test_data['component'] = test_data['component'].map(lambda x: x.lower().split(' '))

    return train_data, test_data

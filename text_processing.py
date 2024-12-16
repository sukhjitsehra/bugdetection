import re
import ast
import wordninja
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Define stop words
stop_words = set(stopwords.words("english"))

def expand_contractions(text, contractions_dict):
    """
    Expand contractions in the given text.
    """
    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contractions_dict.get(match.lower(), match)
        return expanded_contraction

    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())), 
                                       flags=re.IGNORECASE | re.DOTALL)
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def preprocess_text(text, contractions_dict):
    """
    Perform text preprocessing including:
    - Removing punctuation
    - Lowercasing
    - Removing tags and special characters
    - Expanding contractions
    - Tokenizing with WordNinja
    - Lemmatization and stop word removal
    """
    # Remove punctuations and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()

    # Remove tags
    text = re.sub("&lt;/?.*?&gt;", " ", text)

    # Remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    # Expand contractions
    text = expand_contractions(text, contractions_dict)

    # Tokenize
    text = wordninja.split(text)

    # Lemmatize and remove stop words
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

def remove_large_values(lst, size_limit=50000):
    """
    Remove values greater than the specified size_limit from the list.
    """
    lst = ast.literal_eval(lst)  # Convert string to list
    return [int(x) for x in lst if int(x) <= size_limit]

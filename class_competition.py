import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import numpy as np

# Download NLTK resources (run only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the training dataset
dataset_path = "/home/elleixir/Downloads/ling-582-fall-2024-class-competition-code-leliarhodes72/competition-data/train.csv"
dataset = pd.read_csv(dataset_path)
dataset.fillna({'TEXT': ''}, inplace=True)

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Define function for text preprocessing
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        # Convert text to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenization
        tokenizer = TreebankWordTokenizer()
        tokens = tokenizer.tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        # Join tokens back into text
        preprocessed_text = ' '.join(lemmatized_tokens)
        return preprocessed_text
    else:
        return ''  # Return empty string if input is not a string

# Split the text into two spans and preprocess each
def preprocess_spans(text):
    spans = text.split("[SNIPPET]")
    if len(spans) == 2:
        span1, span2 = preprocess_text(spans[0]), preprocess_text(spans[1])
    else:
        span1, span2 = preprocess_text(spans[0]), ''  # Handle missing span
    return span1, span2

# Apply text preprocessing to the 'TEXT' column
dataset[['SPAN1', 'SPAN2']] = dataset['TEXT'].apply(preprocess_spans).apply(pd.Series)

# Concatenate spans to get a single representation for each entry
combined_text = dataset['SPAN1'] + ' ' + dataset['SPAN2']

# Improved TF-IDF vectorizer with adjusted parameters
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,        # Limits the number of features
    ngram_range=(1, 2),       # Includes unigrams and bigrams
    min_df=2,                 # Ignores terms with frequency < 2
    max_df=0.8                # Ignores terms in more than 80% of documents
)
train_tfidf = tfidf_vectorizer.fit_transform(combined_text)

# Define target variable
y_train = dataset['LABEL']

# Compute cosine similarity as an additional feature
span1_tfidf = tfidf_vectorizer.transform(dataset['SPAN1'])
span2_tfidf = tfidf_vectorizer.transform(dataset['SPAN2'])
cos_sim = np.array([cosine_similarity(span1, span2)[0, 0] for span1, span2 in zip(span1_tfidf, span2_tfidf)])

# Combine TF-IDF features with cosine similarity
train_features = hstack([train_tfidf, cos_sim.reshape(-1, 1)])

# Train gradient boosting classifier
gbc = GradientBoostingClassifier()
gbc.fit(train_features, y_train)

# Load and preprocess the test set
test_dataset_path = "/home/elleixir/Downloads/ling-582-fall-2024-class-competition-code-leliarhodes72/competition-data/test.csv"
test_dataset = pd.read_csv(test_dataset_path)
test_dataset.fillna({'TEXT': ''}, inplace=True)
test_dataset[['SPAN1', 'SPAN2']] = test_dataset['TEXT'].apply(preprocess_spans).apply(pd.Series)
test_combined_text = test_dataset['SPAN1'] + ' ' + test_dataset['SPAN2']
test_tfidf = tfidf_vectorizer.transform(test_combined_text)

# Compute cosine similarity for test set
test_span1_tfidf = tfidf_vectorizer.transform(test_dataset['SPAN1'])
test_span2_tfidf = tfidf_vectorizer.transform(test_dataset['SPAN2'])
test_cos_sim = np.array([cosine_similarity(span1, span2)[0, 0] for span1, span2 in zip(test_span1_tfidf, test_span2_tfidf)])

# Combine test TF-IDF features with cosine similarity
test_features = hstack([test_tfidf, test_cos_sim.reshape(-1, 1)])

# Make predictions on the test set
test_predictions = gbc.predict(test_features)

# Save predictions to submission file
test_dataset['LABEL'] = test_predictions
test_dataset[['ID', 'LABEL']].to_csv('submission.csv', index=False)

# Save predictions to submission file
test_dataset['LABEL'] = test_predictions
test_dataset[['ID', 'LABEL']].to_csv('submission.csv', index=False)

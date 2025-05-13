import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# Download NLTK resources (run only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
#dataset_path = ('comment out my path and add your hown to the training data')
dataset_path = "/home/elleixir/HLT/ling-539-sp-2024-class-competition/train.csv" 
dataset = pd.read_csv(dataset_path)

dataset.fillna({'TEXT' : ''}, inplace=True)

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
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        # Join tokens back into text
        preprocessed_text = ' '.join(lemmatized_tokens)
        return preprocessed_text
    else:
        # Return empty string if input is not a string
        return ''


# Apply text preprocessing to the 'TEXT' column
dataset['TEXT'] = dataset['TEXT'].apply(preprocess_text)




tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the 'TEXT' column of the dataset to obtain TF-IDF features
train_tfidf = tfidf_vectorizer.fit_transform(dataset['TEXT'])




# Define target variable
y_train = dataset['LABEL'] 

# Instantiate the logistic regression model
logistic_regression_model = LogisticRegression()

# Train the model on the training data
logistic_regression_model.fit(train_tfidf, y_train)




# Load the test set
#dataset_path = ('comment out my path and add your hown to the training data')
test_dataset_path = "/home/elleixir/HLT/ling-539-sp-2024-class-competition/test.csv" 
test_dataset = pd.read_csv(test_dataset_path)

# Preprocess test data
test_dataset['TEXT'] = test_dataset['TEXT'].apply(preprocess_text)

# Transform test data into TF-IDF features
test_tfidf = tfidf_vectorizer.transform(test_dataset['TEXT'])

# Make predictions on the test set
predictions = logistic_regression_model.predict(test_tfidf)

# Add predicted labels to the test dataset
test_dataset['LABEL'] = predictions

# Save the DataFrame to a CSV file
test_dataset[['ID', 'LABEL']].to_csv('submission.csv', index=False)

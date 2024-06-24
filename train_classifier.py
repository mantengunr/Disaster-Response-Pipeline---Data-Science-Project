import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.externals import joblib


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# Define a function to data
def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath) # Creating a database engine to connect to the SQLite database
    query = "SELECT * FROM DisasterResponse" # Defining a SQL query to select all rows from the 'DisasterResponse' table
    df = pd.read_sql(query, engine) # Reading the data from the 'DisasterResponse' table into a pandas DataFrame
    df = df.dropna() # Dropping any rows with missing values from the DataFrame

    X = df['message'] # Extracting the 'message' column as the feature (X)
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1) # Extracting all columns except 'id', 'message', 'original', and 'genre' as the target variables (Y)
    category_names = Y.columns.tolist() # Getting the list of category names from the target variables
    
    return X,Y,category_names

# Define a function to tokenize text
def tokenize(text):
    detected_urls = re.findall(url_regex, text) # Find all URLs in the text using a regular expression pattern
    
    # Iterate over each detected URL
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder") # Replace the URL with a placeholder string "urlplaceholder"
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Replacing any non-alphanumeric characters with a space and convert the text to lowercase
    tokens = word_tokenize(text) # Tokenizing the text into individual words
    lemmatizer = WordNetLemmatizer() # Creating a WordNetLemmatizer object for lemmatizing the tokens
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens] # Lemmatizeing each token and remove any leading or trailing whitespace
    return clean_tokens

# Define a function to build the model
def build_model():
    # Creating a pipeline for text classification
    pipeline = Pipeline([
        # Step 1: Convert text into a matrix of token counts using CountVectorizer
        ('vect', CountVectorizer(tokenizer=tokenize)),
        # Step 2: Apply TF-IDF (Term Frequency-Inverse Document Frequency) transformation
        ('tfidf', TfidfTransformer()),
        # Step 3: Train a multi-output classifier using Random Forest algorithm
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
     ]) 

    return pipeline

# Define a function to evaluate the performance of a model on the test set
def evaluate_model(model, X_test, Y_test, category_names):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Iterate over each category
    for i in range(len(category_names)):
        # Print the category name
        print(f'Category: {category_names[i]}')
        
        # Generate a classification report for the category
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))

# Define a function to save the trained model to a file
def save_model(model, model_filepath):
    # Save the model using the joblib library
    joblib.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print(model_filepath)
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        params = model.get_params() # Get the parameters of the pipeline

        # Print the parameters
        for param, value in params.items():
            print(param, ":", value)
        
        # Define a dictionary of parameters for grid search
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': [True, False],
            'clf__estimator__n_estimators': [50, 100]

        }

        # Create a GridSearchCV object with the pipeline and parameters
        model = GridSearchCV(model, parameters, cv=2)   
        # Fit the GridSearchCV model on the training data
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
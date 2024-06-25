# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine, inspect, MetaData

def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges the messages and categories datasets, and performs necessary data transformations.

    Args:
        messages_filepath (str): The file path to the messages dataset.
        categories_filepath (str): The file path to the categories dataset.

    Returns:
        pandas.DataFrame: The merged dataframe containing messages and categories.

    """
    messages = pd.read_csv(messages_filepath) # load messages dataset
    categories = pd.read_csv(categories_filepath) # load categories dataset
    categories = categories['categories'].str.split(';', expand=True) # creating a dataframe of the 36 individual category columns
    row = categories.iloc[0] # selecting the first row of the categories dataframe
    category_colnames = row.apply(lambda x: x[:-2]) # useing the selected row to extract a list of new column names for categories.
    categories.columns = category_colnames # rename the columns of `categories`
    
    # Converting category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1]) # setting each value to be the last character of the string
        categories[column] = categories[column].astype(int) # convert column from string to numeric
    df = pd.concat([messages, categories], axis=1)     
    return df

# Cleaning the loaded data
def clean_data(df):
    """
    Cleans the input dataframe by dropping duplicate rows.

    Args:
        df (pandas.DataFrame): The input dataframe to be cleaned.

    Returns:
        pandas.DataFrame: The cleaned dataframe with duplicate rows removed.

    """
    df = df.drop_duplicates() # dropping duplicates
    return df

def save_data(df, database_filename):
    """
    Saves the input dataframe to a SQLite database.

    Args:
        df (pandas.DataFrame): The input dataframe to be saved.
        database_filename (str): The file path of the SQLite database.

    """
    engine = create_engine('sqlite:///'+database_filename) # Creating a database engine
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace') # Saveing a pandas DataFrame (df) to a table named 'DisasterResponse' in the SQLite database

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
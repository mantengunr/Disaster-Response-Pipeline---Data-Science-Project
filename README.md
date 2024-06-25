# Disaster Response Pipeline - Data Science Project

## Project Description
This project focuses on utilizing an Extract, Transform, and Load (ETL) pipeline to process pre-labeled tweets and text messages related to real-life disasters from Appen (formerly Figure 8). The goal is to develop a supervised machine learning model using Natural Language Processing (NLP) techniques to classify these disaster messages. This classification will help disaster response teams to efficiently filter and prioritize relevant messages, ensuring timely responses from various emergency units.

## Repository (Folders and Files)
The repository is comprised of data, models, and app folders.

    - The data folder has raw data (disaster_massegaes.csv and disaster_categories.csv), data wrangling and database creating python script (process_data.py) and the procsses data database (DisasterResponse.db).
    
    - The models folder has python script (train_classifier.py) that get the data from the database and use it to create and evaluate the text classification machine learning model.
    
    - The app folder has a python script (run.py) that runs the web application for text classification and results visualization


Below are the step-by-step instructions to run the ETL & ML Pipeline, and Web application.


## Project Testing (Demo) Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

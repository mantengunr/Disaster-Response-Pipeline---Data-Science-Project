# Disaster Response Pipeline - Data Science Project
This project looks at analyzing disaster data from Appen(formerly Figure 8) to build a model for an API that classifies disaster messages.

The repository is comprised of data, models, and app folders.
    - The data folder has raw data (disaster_massegaes.csv and disaster_categories.csv), data wrangling and database creating python script (process_data.py) and the procsses data database (DisasterResponse).
    - The models folder has python script (train_classifier.py) that get the data from the database and use it to create and evaluate the text classification machine learning model.
    - The app folder has a python script that runs the web application for text classification and results visualization

Below are the step-by-step instructions to run the ETL & ML Pipeline, and Web application.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

# Disaster Response Pipeline Project

## Introduction
This project was created as part of a project evaluation for Udacity.

The goal is to create a ML model to classify messages into categories. 

Frist, the data set are cleaned to prepare the data to be evaluated. 
Second,  a ML is trained using Pipelines.
Finally, the model can be tested through a demo web page

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl [True/False(default)]`

2. Run the following command in the app's directory to run your web app. 
- `cd app`
- `python run.py`

3. Go to http://localhost:3001/

## File Description

- **data_exploring**: A set of Jupyter notebooks with an initial approach
  - **__init__.py**: File to let Python know this is a package
  - **etl_pipeline_preparation.ipynb**: Jupyter notebook to merge and clean the dataseat
  - **ml_pipeline_preparation.ipynb**: Jupyter notebook to fit and evaluate the ML model
  - **test_cv**: Model definition using GripSearch
- **data**: A folder with the data set and the script to clean it up.
  - **disaster_categories.csv**: CSV with the categories
  - **disaster_messages.csv**: CSV with the messages
  - **process_data.py**: Scrip to merge datasets and clean them up
  - **DisasterResponse.db**: SQLite database with the cleaned dataset
- **model**: A folder with script to fit the model with a given SQLite databse and the pickle object with the saved model
  - **train_classifier.py**: Script to fit and evalute the ML model
  - **classifier.pkl** Pickle file with the fitted model
- **app**: Flask app to test the model by user in a browser
  - **templates**: Templates folder
    - **go.html**: HTML for main page
    - **master.html**: Jinga template for used by other templates
  - **run.py**: Flask app main file
- **enviroment.yml**: A export file with the dependencies in Conda format
- **requirements.txt**: A export file with the dependencies in PIP format

## Result Summary

The messages were tokenized using **NLTK**, first the messages are lemmatized, then only the ones with words (not punctuations or symbols) are selected and then, all "stop words" are removed.
The model was created using a **Scikit-Learn**, with a **TF-IDF** approach and classified using **Random Forest**. 

**The result was %90+ accuracy** 
import sys
import pickle

import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    INPUT
    database_filepath - SQLite database Filepath
    OUTPUT
    X - Features
    Y - Output
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("messages", engine)
    X = df.message.values
    Y = df.iloc[:, 4:]
    return X, Y


def tokenize(text):
    '''
    INPUT
    text - Messages

    OUTPUT
    tokens - Cleaned message splited in words

    Description:
    This functions formats the categories in a way they are distributed over the columns
    and clean dups
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t).lower().strip() for t in tokens]
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in set(stopwords.words('english'))]
    return tokens


def build_model(grid_search=False):
    '''
    INPUT
    grid_search - Use Grid Search to find the best model (False default)

    OUTPUT
    cv - Model wrapped in a GridSearch
    '''
    pipeline = Pipeline([

        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    if not grid_search:
        return pipeline

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    '''
    INPUT
    model - ML model
    X_test - Features for testing
    Y_test - Output for testing

    Description:
    This functions fit the model and evalute the prediction
    '''
    Y_pred = model.predict(X_test)

    for i in range(Y_test.shape[1]):
        print("=====", Y_test.columns[i] ,"=====")
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i], output_dict=False, zero_division=0))


def save_model(model, model_filepath):
    '''
    INPUT
    model - ML model
    model_filepath - Pickle Filepath to export model

    '''
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)


def main():
    if len(sys.argv) in (4,5):
        if len(sys.argv) == 4:
            database_filepath, model_filepath = sys.argv[1:]
            grip_search = False
        else:
            database_filepath, model_filepath, grip_search = sys.argv[1:]
            grip_search = grip_search.lower() == "true"

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(grip_search)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl False')


if __name__ == '__main__':
    main()
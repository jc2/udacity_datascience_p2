import sys
import json
import pickle

import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

# Hack to make it work. It is needed a better project structure
sys.path.append("..")
from models.train_classifier import *


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
# model = joblib.load("../models/classifier.pkl")
with open('../models/classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    genres = df.genre.value_counts()
    categories = df.iloc[:, 4:].sum()
    
    graphs = [
        {
            'data': [
                Bar(
                    x=list(genres.index),
                    y=list(genres.values)
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=list(categories.index),
                    y=list(categories.values)
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                },
                'margin': {
                    't': 100,
                    'b': 200
                },
                'height': 600
            }
        },

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
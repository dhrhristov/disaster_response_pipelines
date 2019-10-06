import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
   tokens = word_tokenize(text)
   lemmatizer = WordNetLemmatizer()

   clean_tokens = []
   for tok in tokens:
      clean_tok = lemmatizer.lemmatize(tok).lower().strip()
      clean_tokens.append(clean_tok)

   return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #sorting the results
    sort_bycount_genre_x = [x for _,x in sorted(zip(genre_counts, genre_names))]
    sort_bycount_genre_y = sorted(genre_counts)

    hospitals_counts = df.groupby('hospitals').count()['message']
    hospitals_names = list(genre_counts.index)

    #sorting the results
    sort_bycount_hospitals_x = [x for _,x in sorted(zip(hospitals_counts, hospitals_names))]
    sort_bycount_hospitals_y = sorted(hospitals_counts)    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
	
	# predefined Graph
        #{
        #    'data': [
        #        Bar(
        #            x=genre_names,
        #            y=genre_counts,
        #            text=genre_counts,
        #            textposition = 'auto',
        #            marker={'color': genre_counts,
        #                    'colorscale': 'Jet'}
        #        )
        #    ],
        #
        #
        #    'layout': {
        #        'title': 'Distribution of Message Genres',
        #        'yaxis': {
        #            'title': "Count"
        #        },
        #        'xaxis': {
        #            'title': "Genre"
        #        }
        #    }
        #}


        # Custom Graph based on Genre and Message
        {

            'data': [
                Bar(
                    x=sort_bycount_genre_x,
                    y=sort_bycount_genre_y,
                    text=sort_bycount_genre_y,
                    textposition = 'auto',
                    marker={'color': sort_bycount_genre_y,
                            'colorscale': 'Jet'},
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count per Category: Genre"
                },
                'xaxis': {
                    'title': "Category Name: Genre",
                },
                'height': 600,
                'margin': dict(
                    b = 160, 
                )
            }
         },



         # Custom Graph based on Hospitals and Message
         {
            'data': [
                Bar(
                    x=sort_bycount_hospitals_x,
                    y=sort_bycount_hospitals_y,
                    text=sort_bycount_hospitals_y,
                    textposition = 'auto',
                    marker={'color': sort_bycount_hospitals_y,
                            'colorscale': 'Jet'},
                )
            ],

            'layout': {
                'title': 'Distribution of Message Hospitals',
                'yaxis': {
                    'title': "Count per Category: Hospitals"
                },
                'xaxis': {
                    'title': "Category Name: Hospitals",
                },
                'height': 600,
                'margin': dict(
                    b = 160,
                )
            }
        }
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

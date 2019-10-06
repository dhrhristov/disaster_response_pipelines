import sys
import re
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import FunctionTransformer


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine

def load_data(database_filepath):

   """ 
   Function for Importing Data from Database
   
   input:
        database_filepath: path/filename to SQL database
    	        
   output:
        X: feature variable
	Y: target variable
	category_names:  individual category used for classification
   """

   # Import data from SQL Lite database
   engine = create_engine('sqlite:///{}'.format(database_filepath))
   df = pd.read_sql_table('disaster_messages', engine)
   X = df['message']
   Y = df.iloc[:, 4:]
   category_names = Y.columns

   return X, Y, category_names


def tokenize(text):
    
   """ 
   Function to Tokenize text 
   
   input:
       text: the original text data
    	        
   output:
       tokens: Uses clean text to tokenize and initiate lemmatizer
   """
    
   text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
   # tokenize text
   initial_tokens = word_tokenize(text)

   # initiate lemmatizer
   lemmatizer = WordNetLemmatizer()

   # iterate through each token
   tokens = []
   for token in initial_tokens:
       
       # 1) lemmatize
       # 2) convert to lower case
       # 3) strip white spaces front and back

       processed_token = lemmatizer.lemmatize(token).lower().strip()
       tokens.append(processed_token)

   return tokens   


def build_model():

    """ 
    Function for Creating Model
    Based on the Model is created pipeline and using grid search to find 
    optimal parameters.
    
    input:
       None
    output: 
       sklearn model
    """

    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
	
    # Provided Parameters are adjusted based on experiments to not take too
    # much training time

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100]
    }
    
    cv = GridSearchCV(pipeline, parameters)

    return cv


def evaluate_model(model, X_test, Y_test, categories):

    """ 
    Function for evaluating the Model. 
    
    input:
       model: sklearn model created by build_model()
       X_test: the test data set
       Y_test: the set of labels to all the data in x_test
       categories:  individual category used for classification

    output: 
       Generate classification Report 
    """

    Y_pred = model.predict(X_test)

    # Calculate the accuracy for every category
    for i in range(len(categories)):
        print("Category:", categories[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(categories[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    
    """ 
    Export your model as a pickle file
    
    input:
       model which will be saved in a pickled serialization
       filename which sets the name for the pickle file
    	        
    output:
        Function does not return directly {model}.pkl file
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    
    """

    The application expect 2 arguments:
    1) filepath to database
    2) filename to store pickle file with sklearn model

    """

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
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

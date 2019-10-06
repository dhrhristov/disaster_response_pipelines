# Disaster Response Pipeline Project


### Installations
In order to run the codes in this project, the following libraries must be installed:
	1.	Pandas
	2.	Plotly
	3.	Numpy
	4.	Sci-kit Learn (for model creation)
	5.	pickle (for model storage)
	6.	SQL Alchemy (for SQL Processing)
	7.	NLTK (stem, tokenize)
	8.	Flask  (for web app)

### Motivation
Generate a Web application which can query Emergency message processing and analysis. Provide visual results for analysis. It is part of Udacity's Data Scientist Nanodegree. 

### Folder structure
The project contains  3 folders: 
    1.    DATA: contains the SQL data and data processing logic
    2.   MODEL: trained Model and the script used to generate and evaluate the model.
    3.  APP: contains the web app which is used to get new queries and do visualization 
Files in the Data Folder
	1.	Messages data: disaster_messages.csv
	2.	Categories data: disaster_categories.csv
	3.	SQL Database: DisasterResponse.db
    4.     Python script for processing the data: process_data.py

Files in the Models Folder
    1.     Python script for training the classifier: train_classifier.py
    2.     A pickle generated file that contains the trained model: classifier.pkl

Files in the App Folder
	1.	Python script for running the web app: run.py
	2.	templates folder that contains 2 HTML files for the app front-end: go.html and master.html

### Instructions how to use application:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### What to check when Application is started 
A web page provides user interface to query a database and visualize the query in graphs for analysis based on trained model.
1.   Check the visual graphs
2.   Check the query Analysis
 

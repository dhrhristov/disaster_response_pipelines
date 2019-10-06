import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """ 
    input:
    	messages_filepath: messages dataset CSV file name with path
    	categories_filepath: categories dataset CSV file name with path
    	        
    output:
		1. read two datasets
		2. merge both datasets based on 'id'
		3. return pandas dataset 'df'
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    
    # this function returns df 
    return df


def clean_data(df):

    """ 
    input:
    	df: merged dataset from messages and categories
    	        
    output:
	df where categories columns are renamed
    """
    
    # extract the individual category column names based on first row
    categories = df.categories.str.split(pat=';', expand=True)
    row = categories.iloc[0]
   
    category_colnames = row.apply(lambda x:x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    	#set each value to be the last character of the string
    	categories[column] = categories[column].str[-1]
    	
    	# convert column from string to numeric
    	categories[column] = categories[column].astype(int)
    	
    # Remove the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # Merge the original dataframe with the renamed `categories`
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):

    """  
    Write records stored in a Data Frame to a SQL database.

    input:
    	             df: dataset which will be started in SQL database
      database_filename: database name
    	         
    output:
      Saves the data frame into sqlite database. 

    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_messages', engine, index=False)
    
 


def main():
    
    """

    The application expect 3 arguments:
    1) filename to messages CSV file database
    2) filename to categories CSV file database
    3) filename to pickle file with sklearn model

    """
    
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

import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - Messages Filepath
    categories_filepath - Categories Filepath

    OUTPUT
    df - Merged DataFrame
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')
    return df


def clean_data(df):
    '''
    INPUT
    df - Merged Datafame

    OUTPUT
    df - Cleaned DataFrame

    Description:
    This functions formats the categories in a way they are distributed over the columns
    and clean dups
    '''
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x.split("-")[1]))

    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)

    df = df.drop_duplicates(subset=["message"])

    df = df[df["related"] != 2]

    return df


def save_data(df, database_filename):
    '''
    INPUT
    df - Cleaned Datafame
    database_filename - SQLite Filepath

    OUTPUT
    None

    Description:
    This function saves DataFrame into a SQLite table and store it
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False)  


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
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories datasets from the specified filepaths
    Args:
        messages_filepath: Filepath to the messages dataset
        categories_filepath: Filepath to the categories dataset
    Returns:
        (DataFrame) df: Merged Pandas dataframe
    """
    messages = pd.read_csv(messages_filepath) # Read messages
    categories = pd.read_csv(categories_filepath) # Read categories
    df = messages.merge(categories, on='id') # Merge messages and categories

    return df


def clean_data(df):
    """
    Cleans the merged dataset
    Args:
        df: Merged pandas dataframe
    Returns:
        (DataFrame) df: Cleaned dataframe
    """
    categories = df['categories'].str.split(';',expand=True)
    first_row = categories.iloc[0]
    category_colnames = first_row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert category values to numbers (0/1)
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    # drop the original categories column
    df.drop('categories',axis=1,inplace=True)

    # perform concatenation of the original dataframe with the new 'categories' dataframe
    df = pd.concat([df,categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates() # default: keep='first'

    return df


def save_data(df, database_filename):
    """
    Saves clean dataset into an sqlite database
    Args:
        df:  Cleaned dataframe
        database_filename: Name of the database file
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('cleanData', engine, index=False)


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

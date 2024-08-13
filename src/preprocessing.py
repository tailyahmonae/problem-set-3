'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    
    model_pred_df = pd.read_csv('prediction_model_03.csv')
    genres_df = pd.read_csv('genres.csv')
    
    return model_pred_df, genres_df

def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    # extract unique genres
    genre_list = genres_df['genre'].unique().tolist()   
    
    # initialize counters
    genre_true_counts = {genre: 0 for genre in genre_list}
    genre_tp_counts = {genre: 0 for genre in genre_list}
    genre_fp_counts = {genre: 0 for genre in genre_list}
    
    for _, row in model_pred_df.iterrows():
        actual_genres = row['actual genres'].strip("[]").replace("'", "").split(', ')
        # predicted genre
        predicted_genre = row['predicted']
        # update counts for actual genres
        for genre in actual_genres:
            genre_true_counts[genre] += 1
            if predicted_genre in actual_genres:
                genre_tp_counts[predicted_genre] += 1
            else:
                genre_fp_counts[predicted_genre] += 1

    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
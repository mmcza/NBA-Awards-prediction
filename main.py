from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import pickle
from award_prediction import prediction_to_dict
import ast

def main():
    # Check if the model exists
    if not Path('models/all_nba_pred_model.pkl').exists():
        print('Model does not exist. Please train the model first.')
        return {}
    if not Path('models/all_rookie_pred_model.pkl').exists():
        print('Model does not exist. Please train the model first.')
        return {}
    # Read argument
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    results_file = Path(args.results_file)

    # Load data
    seasonal_stats = pd.read_csv('data/seasonal_stats_with_awards.csv')
    regular_seasons = seasonal_stats[seasonal_stats['MATCH_TYPE'] == 'Regular']
    def convert_season_to_start_year(season):
        return int(season[:4])

    regular_seasons['SEASON_START'] = regular_seasons['SEASON'].apply(convert_season_to_start_year)
    regular_seasons = regular_seasons.loc[regular_seasons['SEASON_START'] == 2023]

    # Calculate per game statistics
    regular_seasons['MIN_per_GP'] = regular_seasons['MIN'] / regular_seasons['GP']
    regular_seasons['PTS_per_GP'] = regular_seasons['PTS'] / regular_seasons['GP']
    regular_seasons['REB_per_GP'] = regular_seasons['REB'] / regular_seasons['GP']
    regular_seasons['AST_per_GP'] = regular_seasons['AST'] / regular_seasons['GP']
    regular_seasons['STL_per_GP'] = regular_seasons['STL'] / regular_seasons['GP']
    regular_seasons['BLK_per_GP'] = regular_seasons['BLK'] / regular_seasons['GP']
    regular_seasons['TO_per_GP'] = regular_seasons['TO'] / regular_seasons['GP']
    regular_seasons['FP_per_GP'] = regular_seasons['FP'] / regular_seasons['GP']
    regular_seasons['PIE_per_GP'] = regular_seasons['PIE'] / regular_seasons['GP']
    regular_seasons['FGM_per_GP'] = regular_seasons['FGM'] / regular_seasons['GP']
    regular_seasons['FGA_per_GP'] = regular_seasons['FGA'] / regular_seasons['GP']
    regular_seasons['FG3M_per_GP'] = regular_seasons['FG3M'] / regular_seasons['GP']
    regular_seasons['FG3A_per_GP'] = regular_seasons['FG3A'] / regular_seasons['GP']
    regular_seasons['FTM_per_GP'] = regular_seasons['FTM'] / regular_seasons['GP']
    regular_seasons['FTA_per_GP'] = regular_seasons['FTA'] / regular_seasons['GP']

    # Impute missing values for 3PT percentage
    regular_seasons['FG3_PCT'] = regular_seasons['FG3_PCT'].fillna(0)

    # Separate rookies
    rookies = regular_seasons.loc[regular_seasons['ROOKIE_SEASON'] == 1]

    # Filter players
    regular_seasons = regular_seasons.loc[regular_seasons['GP'] >= 40]
    regular_seasons = regular_seasons.loc[regular_seasons['MIN'] >= 1250]
    regular_seasons = regular_seasons.loc[regular_seasons['PTS'] >= 333]
    regular_seasons = regular_seasons.loc[regular_seasons['FP'] >= 1250]
    rookies = rookies.loc[rookies['GP'] >= 24]
    rookies = rookies.loc[rookies['MIN'] >= 650]
    rookies = rookies.loc[rookies['PTS'] >= 250]
    rookies = rookies.loc[rookies['FP'] >= 500]

    # Normalize the data
    normalize_columns = ['MIN_per_GP', 'PTS_per_GP', 'REB_per_GP', 'AST_per_GP', 'STL_per_GP', 'BLK_per_GP',
                         'TO_per_GP', 'FP_per_GP', 'PIE_per_GP', 'FGM_per_GP', 'FGA_per_GP', 'FG3M_per_GP',
                         'FG3A_per_GP', 'FTM_per_GP', 'FTA_per_GP', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FP',
                         'PIE', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'MIN', 'GP', 'W']
    normalized_season2324 = regular_seasons.copy()
    normalized_rookies2324 = rookies.copy()
    max2324 = regular_seasons.max()
    max_rookies2324 = rookies.max()
    for column in normalize_columns:
        normalized_season2324[column] = regular_seasons[column] / max2324[column]
        normalized_rookies2324[column] = rookies[column] / max_rookies2324[column]

    # Get player names and MVP status
    names_2324 = normalized_season2324['PLAYER_NAME'].to_numpy()
    mvp_2324 = normalized_season2324['PLAYER_NAME'][normalized_season2324['MVP'] == 1].values[0]
    rookie_names_2324 = normalized_rookies2324['PLAYER_NAME'].to_numpy()
    roy_2324 = normalized_rookies2324['PLAYER_NAME'][normalized_rookies2324['ROY'] == 1].values[0]

    # Drop unnecessary columns
    columns_to_drop = ['SEASON', 'SEASON_START', 'MATCH_TYPE', 'All-NBA-Team', 'PLAYER_NAME',
                        'All-Defensive-Team', 'All-Rookie-Team', 'Finals-MVP', 'PLAYER_ID',
                       'MVP', 'ROY', '6MOY', 'MIP', 'ROOKIE_SEASON', 'FTM', 'FGM', 'FG3M']
    normalized_season2324 = normalized_season2324.drop(columns=columns_to_drop)
    normalized_rookies2324 = normalized_rookies2324.drop(columns=columns_to_drop)

    # Load models
    with open('./models/all_nba_pred_model.pkl', 'rb') as f:
        all_nba_model = pickle.load(f)

    with open('./models/all_rookie_pred_model.pkl', 'rb') as f:
        all_rookie_model = pickle.load(f)

    # Load information about used features
    features_all_nba = pd.read_csv('models/model_scores.csv')
    features_all_rookie = pd.read_csv('models/model_scores_rookie.csv')

    # Sort features by score and select the best ones and if score is the same, sort by the first column ascending
    features_all_nba = features_all_nba.sort_values(by=['score', 'Unnamed: 0'], ascending=[False, True])
    features_all_rookie = features_all_rookie.sort_values(by=['score', 'Unnamed: 0'], ascending=[False, True])
    best_predictor_all_nba = features_all_nba.head(1)
    best_predictor_all_rookie = features_all_rookie.head(1)

    # Get the used features as a list
    used_statistics_all_nba = best_predictor_all_nba['used_statistics'].values[0]
    used_statistics_all_nba = used_statistics_all_nba.replace(" ", ",")
    features_used_all_nba = ast.literal_eval(used_statistics_all_nba)
    used_statistics_rookie = best_predictor_all_rookie['used_statistics'].values[0]
    used_statistics_rookie = used_statistics_rookie.replace(" ", ",")
    features_used_rookie = ast.literal_eval(used_statistics_rookie)

    # Make the predictions
    all_nba_predictions = prediction_to_dict(all_nba_model.predict_proba(normalized_season2324[features_used_all_nba]),
                                             names_2324, mvp_2324, voting=best_predictor_all_nba['vote'].values[0], rookie=False)
    all_rookie_predictions = prediction_to_dict(all_rookie_model.predict_proba(normalized_rookies2324[features_used_rookie]),
                                                rookie_names_2324, roy_2324, voting=best_predictor_all_rookie['vote'].values[0], rookie=True)

    # Save the predictions
    prediction = {}
    for key, value in all_nba_predictions.items():
        prediction[key] = value
    for key, value in all_rookie_predictions.items():
        prediction[key] = value

    # Save the predictions to a file
    with open(results_file, 'w') as f:
        json.dump(prediction, f)

if __name__ == '__main__':
    main()
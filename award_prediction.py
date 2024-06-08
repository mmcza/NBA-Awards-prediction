import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Define the metrics for the prediction
def award_metrics(prediction, real_values, rookie=False):
    points = 0
    correct_teams = {1: 0, 2: 0, 3: 0}
    if rookie:
        num_teams = 2
        team_names = {1: "first rookie", 2: "second rookie"}
    else:
        num_teams = 3
        team_names = {1: "first", 2: "second", 3: "third"}

    # Calculate points
    for team in range(1, num_teams + 1):
        for player in prediction[f'{team_names[team]} all-nba team']:
            if player in real_values[f'{team_names[team]} all-nba team']:
                points += 10
                correct_teams[team] += 1
            if team < 3 and player in real_values[f'{team_names[team + 1]} all-nba team']:
                points += 8
            if not rookie:
                if team < 2 and player in real_values[f'{team_names[team + 2]} all-nba team']:
                    points += 6
            if team > 1 and player in real_values[f'{team_names[team - 1]} all-nba team']:
                points += 8
            if not rookie:
                if team > 2 and player in real_values[f'{team_names[team - 2]} all-nba team']:
                    points += 6

    # Calculate bonus points
    for key, value in correct_teams.items():
        if value == 5:
            points += 40
        elif value == 4:
            points += 20
        elif value == 3:
            points += 10
        elif value == 2:
            points += 5

    return points

# Function to sum up points for prediction of each player
def award_voting(prediction):
    points = np.ones_like(prediction)
    for i in range(len(prediction)):
        points[i, :] *= prediction[i, 1]*5 + prediction[i, 2]*3 + prediction[i, 3]*1

    return points

# Function to convert the prediction to a dictionary
def prediction_to_dict(prediction, names, mvp, rookie=False, voting=False):
    if rookie:
        prediction_dict = {"first rookie all-nba team": [],
                           "second rookie all-nba team": []}
        players = []
        team_names = {1: "first rookie", 2: "second rookie"}
        top_players = 10
        num_teams = 2
    else:
        prediction_dict = {"first all-nba team": [mvp],
                            "second all-nba team": [],
                            "third all-nba team": []}
        players = [mvp]
        team_names = {1: "first", 2: "second", 3: "third"}
        top_players = 15
        num_teams = 3

    pred_np = np.array(prediction)
    if voting:
        pred_np = award_voting(pred_np)
    top_predictions = {}
    # Get the players with highest probability for each team
    for team in range(1, num_teams + 1):
        sorted_indices = np.argsort(pred_np[:, team])[::-1]
        top_indices = sorted_indices[:top_players]
        top_predictions[team] = top_indices

    # Add the players to the dictionary
    for team in range(1, num_teams + 1):
        for player in top_predictions[team]:
            if names[player] not in players:
                players.append(names[player])
                prediction_dict[f"{team_names[team]} all-nba team"].append(names[player])
            if len(prediction_dict[f"{team_names[team]} all-nba team"]) == 5:
                break

    return prediction_dict

# Function to calculate the score of the model
def calculate_score(model, validation_set, columns, vote=True):
    score = 0
    for season in validation_set:
        prediction = prediction_to_dict(model.predict_proba(validation_set[season]['data'][columns]), validation_set[season]['names'],
                                         validation_set[season]['mvp'], voting=vote)
        real_nba_teams = validation_set[season]['real_nba_teams']
        score += award_metrics(prediction, real_nba_teams)
    return score/4

# Function to randomly select parameters for the models
def randomize_parameters():

    # Create parameter grid for the models
    param_grid = {'RFC': {'n_estimators': [100, 200, 300, 400, 500],
                          'max_depth': [10, 25, 50, 100, None],
                          'min_samples_split': [2, 5, 10],
                          'min_samples_leaf': [1, 2, 4],
                          'max_features': ['sqrt', 'log2']},
                    'XGB': {'n_estimators': [100, 200, 300, 400, 500],
                            'max_depth': [10, 25, 50, 100, None],
                            'learning_rate': [0.01, 0.05, 0.1, 0.2],
                            'subsample': [0.6, 0.8, 1],
                            'colsample_bytree': [0.5, 0.8, 1],
                            'gamma': [0, 0.1, 0.2, 0.3, 0.4]},
                  'LGB': {'n_estimators': [100, 200, 300, 400, 500],
                             'max_depth': [10, 25, 50, 100, None],
                            'learning_rate': [0.01, 0.05, 0.1, 0.2],
                            'subsample': [0.6, 0.8, 1],
                            'colsample_bytree': [0.5, 0.8, 1]},
                  'Voting': {'weights_nr': [0, 1, 2, 3, 4, 5]}
                    }

    # Randomly select parameters
    randomized_params = {}
    for model_name, params in param_grid.items():
        randomized_params[model_name] = {}
        for param, values in params.items():
            randomized_params[model_name][param] = np.random.choice(values)
    return randomized_params

# Function to define the models
def define_models(randomized_params, voting_weights):
    # Random Forest Classifier
    RFC = RandomForestClassifier(n_estimators=randomized_params['RFC']['n_estimators'],
                                 max_depth=randomized_params['RFC']['max_depth'],
                                 min_samples_split=randomized_params['RFC']['min_samples_split'],
                                 min_samples_leaf=randomized_params['RFC']['min_samples_leaf'],
                                 max_features=randomized_params['RFC']['max_features'],
                                 random_state=42)
    # XGBoost Classifier
    XGB = xgb.XGBClassifier(n_estimators=randomized_params['XGB']['n_estimators'],
                            max_depth=randomized_params['XGB']['max_depth'],
                            learning_rate=randomized_params['XGB']['learning_rate'],
                            subsample=randomized_params['XGB']['subsample'],
                            colsample_bytree=randomized_params['XGB']['colsample_bytree'],
                            random_state=42)
    # LightGBM Classifier
    LGB = lgb.LGBMClassifier(n_estimators=randomized_params['LGB']['n_estimators'],
                             max_depth=randomized_params['LGB']['max_depth'],
                             learning_rate=randomized_params['LGB']['learning_rate'],
                             subsample=randomized_params['LGB']['subsample'],
                             colsample_bytree=randomized_params['LGB']['colsample_bytree'],
                             n_jobs=-1,
                             random_state=42)
    # Voting Classifier
    models_for_voting = [('RFC', RFC), ('XGB', XGB), ('LGB', LGB)]
    voting = VotingClassifier(models_for_voting, voting='soft',
                              weights=voting_weights[randomized_params['Voting']['weights_nr']])

    # Create dictionary with models
    models = {'RFC': RFC, 'XGB': XGB, 'LGB': LGB, 'Voting': voting}

    return models

def all_nba_training():
    # Load the data
    seasonal_stats = pd.read_csv('data/seasonal_stats_with_awards.csv')
    # Filter the data to only include regular season games
    regular_seasons = seasonal_stats[seasonal_stats['MATCH_TYPE'] == 'Regular']

    def convert_season_to_start_year(season):
        return int(season[:4])

    regular_seasons['SEASON_START'] = regular_seasons['SEASON'].apply(convert_season_to_start_year)
    regular_seasons = regular_seasons.loc[regular_seasons['SEASON_START'] >= 1988]

    # Add a column to determine if player was in any of the All-NBA teams
    regular_seasons['ANY_ALL_NBA'] = regular_seasons['All-NBA-Team'].apply(lambda x: 1 if x > 0 else 0)
    # Add a column to determine if player was in any of the All-Rookie teams
    regular_seasons['ANY_ALL_ROOKIE'] = regular_seasons['All-Rookie-Team'].apply(lambda x: 1 if x > 0 else 0)

    fig, axs = plt.subplots(2, 2)

    # Plot data on each subplot
    axs[0, 0].scatter(regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'PLAYER_NAME'],
                      regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'GP'])
    axs[0, 0].set_title('Games Played by All-NBA Players')

    axs[0, 1].scatter(regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'PLAYER_NAME'],
                      regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'MIN'])
    axs[0, 1].set_title('Minutes Played by All-NBA Players')

    axs[1, 0].scatter(regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'PLAYER_NAME'],
                      regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'PTS'])
    axs[1, 0].set_title('Points Scored by All-NBA Players')

    axs[1, 1].scatter(regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'PLAYER_NAME'],
                      regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'FP'])
    axs[1, 1].set_title('Fantasy Points by All-NBA Players')

    # Display the plot
    plt.tight_layout()
    plt.show()

    # After plotting, we can see that there can be added some filters to the data
    regular_seasons_len = len(regular_seasons)
    regular_seasons = regular_seasons.loc[regular_seasons['GP'] >= 40]
    regular_seasons = regular_seasons.loc[regular_seasons['MIN'] >= 1250]
    regular_seasons = regular_seasons.loc[regular_seasons['PTS'] >= 333]
    regular_seasons = regular_seasons.loc[regular_seasons['FP'] >= 1250]
    regular_seasons_len_after = len(regular_seasons)
    print(f'Number of rows before filtering: {regular_seasons_len} and after filtering: {regular_seasons_len_after}')

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

    # Separate the 2023-24 season as it is the season we want to predict
    season2324 = regular_seasons.loc[regular_seasons['SEASON_START'] == 2023]
    season2324 = season2324.loc[season2324['GP'] >= 65]
    regular_seasons = regular_seasons.loc[regular_seasons['SEASON_START'] < 2023]
    print(f'Number of players that possibly can be in All-NBA Team for 2023-24 {len(season2324)}')

    # Normalize the data for each season
    seasons = regular_seasons['SEASON_START'].unique()
    seasons_max = regular_seasons.groupby('SEASON_START').max()
    normalize_columns = ['MIN_per_GP', 'PTS_per_GP', 'REB_per_GP', 'AST_per_GP', 'STL_per_GP', 'BLK_per_GP',
                         'TO_per_GP', 'FP_per_GP', 'PIE_per_GP', 'FGM_per_GP', 'FGA_per_GP', 'FG3M_per_GP',
                         'FG3A_per_GP', 'FTM_per_GP', 'FTA_per_GP', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FP',
                         'PIE', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'MIN', 'GP', 'W']
    normalized_seasons = regular_seasons.copy()
    for season in seasons:
        for column in normalize_columns:
            normalized_seasons.loc[normalized_seasons['SEASON_START'] == season, column] /= seasons_max.loc[season, column]

    # Normalize the data for the 2023-24 season
    normalized_season2324 = season2324.copy()
    max2324 = season2324.max()
    for column in normalize_columns:
        normalized_season2324[column] /= max2324[column]

    # Create correlation matrix
    corr_features = ['W', 'MIN_per_GP', 'PTS_per_GP', 'REB_per_GP', 'AST_per_GP', 'STL_per_GP', 'BLK_per_GP',
                     'FP_per_GP', 'PIE_per_GP', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'FGM_per_GP', 'FTM_per_GP',
                     'FG3M_per_GP', 'DD', 'TD', 'All-NBA-Team', 'ANY_ALL_NBA']
    correlation_matrix = normalized_seasons[corr_features].corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()

    # Correlation of awards and All-NBA team selection
    awards = ['MVP', 'DPOY', 'ROY', '6MOY', 'MIP', 'All-Star', 'All-Star-MVP', 'POTW', 'POTM']
    # for award in awards:
    #     print(award)
    #     print(normalized_seasons['All-NBA-Team'].loc[normalized_seasons[award] > 0].value_counts())

    # Divide the data into training and validation sets
    np.random.seed(42)
    val_seasons = np.random.randint(1988, 2023, 4)
    X_train = normalized_seasons.loc[~normalized_seasons['SEASON_START'].isin(val_seasons)]
    X_val = normalized_seasons.loc[normalized_seasons['SEASON_START'].isin(val_seasons)]
    y_train = X_train['All-NBA-Team']

    # Drop unnecessary columns
    columns_to_drop = ['SEASON', 'SEASON_START', 'MATCH_TYPE', 'All-NBA-Team', 'ANY_ALL_NBA', 'PLAYER_NAME',
                        'ANY_ALL_ROOKIE', 'All-Defensive-Team', 'All-Rookie-Team', 'Finals-MVP', 'PLAYER_ID',
                       'MVP', 'ROY', '6MOY', 'MIP', 'ROOKIE_SEASON', 'FTM', 'FGM', 'FG3M']
    not_per_game_stats_to_drop = ['PIE', 'FP', 'PTS', 'FGM_2', 'FTA', 'FGA', 'FTM_2', 'AST', 'REB',
                                  'TO', 'STL', 'MIN', 'BLK', 'FG3A', 'FG3M_2', 'ROTM']
    per_game_stats_to_drop = ['PIE_per_GP', 'FP_per_GP', 'PTS_per_GP', 'FGM_per_GP', 'FTA_per_GP', 'FGA_per_GP',
                              'FTM_per_GP', 'AST_per_GP', 'REB_per_GP', 'TO_per_GP', 'STL_per_GP', 'BLK_per_GP',
                              'MIN_per_GP', 'FG3A_per_GP', 'FG3M_per_GP']
    X_train = X_train.drop(columns_to_drop, axis=1)
    # X_train = X_train.drop(not_per_game_stats_to_drop, axis=1)
    # X_train = X_train.drop(per_game_stats_to_drop, axis=1)

    # Create validation set
    validation_set = {}
    for season in val_seasons:
        validation_set[season] = {}
        season_data = X_val.loc[X_val['SEASON_START'] == season]
        # Save the names of the players
        validation_set[season]['names'] = season_data['PLAYER_NAME'].to_numpy()
        # Save the real All-NBA teams and MVP
        validation_set[season]['real_nba_teams'] = {"first all-nba team": season_data['PLAYER_NAME'][season_data['All-NBA-Team'] == 1].values,
                                                    "second all-nba team": season_data['PLAYER_NAME'][season_data['All-NBA-Team'] == 2].values,
                                                    "third all-nba team": season_data['PLAYER_NAME'][season_data['All-NBA-Team'] == 3].values}
        validation_set[season]['mvp'] = season_data['PLAYER_NAME'][season_data['MVP'] == 1].values[0]
        # Drop unnecessary columns and save the data
        season_data = season_data.drop(columns_to_drop, axis=1)
        validation_set[season]['data'] = season_data

    # Create a dataframe to store the scores of the models
    model_scores = pd.DataFrame(columns=['model_name', 'param', 'used_statistics', 'vote', 'score'])

    # Define the voting weights
    voting_weights = [[1, 1, 1], [1, 2, 1], [1, 1, 2],
                    [2, 1, 1], [2, 2, 1], [1, 2, 2]]
    # Get available columns
    columns = X_train.columns

    # Create a loop to test different configuarations of the models and data
    for i in range(50):
        # Randomly select columns to use
        columns_to_use = np.random.choice(columns, size=np.random.randint(5, len(columns)), replace=False)

        # Create a loop to test different models with different parameters
        for j in range(50):
            print(f'Iteration: {i}/{j}')
            # Randomize the parameters
            randomized_params = randomize_parameters()
            # Define the models
            models = define_models(randomized_params, voting_weights)
            # Train the models
            for model_name, model in models.items():
                model.fit(X_train[columns_to_use], y_train)
                score_vote = calculate_score(model, validation_set, columns_to_use, vote=True)
                score_no_vote = calculate_score(model, validation_set, columns_to_use, vote=False)
                if score_vote>score_no_vote:
                    score = score_vote
                    vote = True
                else:
                    score = score_no_vote
                    vote = False
                new_row = pd.DataFrame({'model_name': [model_name],
                                        'param': [randomized_params[model_name]],
                                        'used_statistics': [columns_to_use],
                                        'score': [score],
                                        'vote': [vote]})
                model_scores = pd.concat([model_scores, new_row], ignore_index=True)
        # Save the statistics of the models as a csv file
        model_scores.to_csv('./models/model_scores.csv')

    # Select the best model
    best_model = model_scores.loc[model_scores['score'].idxmax()]
    print(best_model)
    # Create the best predictor
    if best_model['model_name'] == 'Voting':
        best_predictor = VotingClassifier([('RFC', RandomForestClassifier(n_estimators=best_model['param']['RFC']['n_estimators'],
                                        max_depth=best_model['param']['RFC']['max_depth'],
                                        min_samples_split=best_model['param']['RFC']['min_samples_split'],
                                        min_samples_leaf=best_model['param']['RFC']['min_samples_leaf'],
                                        max_features=best_model['param']['RFC']['max_features'],
                                        random_state=42)),
                                          ('XGB', xgb.XGBClassifier(n_estimators=best_model['param']['XGB']['n_estimators'],
                                        max_depth=best_model['param']['XGB']['max_depth'],
                                        learning_rate=best_model['param']['XGB']['learning_rate'],
                                        subsample=best_model['param']['XGB']['subsample'],
                                        colsample_bytree=best_model['param']['XGB']['colsample_bytree'],
                                        random_state=42)),
                                          ('LGB', lgb.LGBMClassifier(n_estimators=best_model['param']['LGB']['n_estimators'],
                                        max_depth=best_model['param']['LGB']['max_depth'],
                                        learning_rate=best_model['param']['LGB']['learning_rate'],
                                        subsample=best_model['param']['LGB']['subsample'],
                                        colsample_bytree=best_model['param']['LGB']['colsample_bytree'],
                                        random_state=42))],
                                        voting='soft',
                                        weights=voting_weights[best_model['param']['Voting']['weights_nr']])
    elif best_model['model_name'] == 'RFC':
        best_predictor = RandomForestClassifier(n_estimators=best_model['param']['n_estimators'],
                                        max_depth=best_model['param']['max_depth'],
                                        min_samples_split=best_model['param']['min_samples_split'],
                                        min_samples_leaf=best_model['param']['min_samples_leaf'],
                                        max_features=best_model['param']['max_features'],
                                        random_state=42)
    elif best_model['model_name'] == 'XGB':
        best_predictor = xgb.XGBClassifier(n_estimators=best_model['param']['n_estimators'],
                                        max_depth=best_model['param']['max_depth'],
                                        learning_rate=best_model['param']['learning_rate'],
                                        subsample=best_model['param']['subsample'],
                                        colsample_bytree=best_model['param']['colsample_bytree'],
                                        random_state=42)
    elif best_model['model_name'] == 'LGB':
        best_predictor = lgb.LGBMClassifier(n_estimators=best_model['param']['n_estimators'],
                                        max_depth=best_model['param']['max_depth'],
                                        learning_rate=best_model['param']['learning_rate'],
                                        subsample=best_model['param']['subsample'],
                                        colsample_bytree=best_model['param']['colsample_bytree'],
                                        random_state=42)

    # Train the best model
    best_predictor.fit(X_train[best_model['used_statistics']], y_train)

    # Save the model
    with open('./models/all_nba_pred_model.pkl', 'wb') as f:
        pickle.dump(best_predictor, f)

    # Predict the All-NBA teams for the 2023-24 season
    names_2324 = normalized_season2324['PLAYER_NAME'].to_numpy()
    mvp_2324 = normalized_season2324['PLAYER_NAME'][normalized_season2324['MVP'] == 1].values[0]
    predictions_2324 = prediction_to_dict(best_predictor.predict_proba(normalized_season2324[best_model['used_statistics']]), names_2324, mvp_2324, voting=True)
    print("Prediction for 2023/24")
    print(predictions_2324)

if __name__ == '__main__':
    all_nba_training()
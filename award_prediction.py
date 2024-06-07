import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


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

def award_voting(prediction):
    points = np.ones_like(prediction)
    for i in range(len(prediction)):
        points[i, :] *= prediction[i, 1]*5 + prediction[i, 2]*3 + prediction[i, 3]*1

    return points


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
    X_train = X_train.drop(columns_to_drop, axis=1)
    X_train = X_train.drop(not_per_game_stats_to_drop, axis=1)

    # Train the Random Forest Classifier model for all All-NBA teams
    RFC_model = RandomForestClassifier(n_estimators=100, random_state=42)
    RFC_model.fit(X_train, y_train)

    # Predict the All-NBA teams for the validation seasons
    score = 0
    for season in val_seasons:
        season_data = X_val.loc[X_val['SEASON_START'] == season]
        names = season_data['PLAYER_NAME'].to_numpy()
        real_nba_teams = {"first all-nba team": season_data['PLAYER_NAME'][season_data['All-NBA-Team'] == 1].values,
                          "second all-nba team": season_data['PLAYER_NAME'][season_data['All-NBA-Team'] == 2].values,
                          "third all-nba team": season_data['PLAYER_NAME'][season_data['All-NBA-Team'] == 3].values}
        mvp = season_data['PLAYER_NAME'][season_data['MVP'] == 1].values[0]
        season_data = season_data.drop(columns_to_drop, axis=1)
        season_data = season_data.drop(not_per_game_stats_to_drop, axis=1)
        prediction = prediction_to_dict(RFC_model.predict_proba(season_data), names, mvp, voting=True)
        print(f'Season: {season}')
        print(prediction)
        print(real_nba_teams)
        print(award_metrics(prediction, real_nba_teams))
        score += award_metrics(prediction, real_nba_teams)

    print(f'Total score: {score}')
    print(f'Average score: {score / 4}')

    # Calculate the feature importances
    feature_importances = RFC_model.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({'feature': features, 'importance': feature_importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Feature Importances')
    plt.show()

    # Predict the All-NBA teams for the 2023-24 season
    names_2324 = normalized_season2324['PLAYER_NAME'].to_numpy()
    mvp_2324 = normalized_season2324['PLAYER_NAME'][normalized_season2324['MVP'] == 1].values[0]
    normalized_season2324 = normalized_season2324.drop(columns_to_drop, axis=1)
    normalized_season2324 = normalized_season2324.drop(not_per_game_stats_to_drop, axis=1)
    predictions_2324 = prediction_to_dict(RFC_model.predict_proba(normalized_season2324), names_2324, mvp_2324, voting=True)
    print("Prediction for 2023/24")
    print(predictions_2324)

if __name__ == '__main__':
    all_nba_training()
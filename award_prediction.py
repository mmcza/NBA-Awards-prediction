import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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
                     'FP_per_GP', 'PIE_per_GP', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'All-NBA-Team', 'ANY_ALL_NBA']
    correlation_matrix = normalized_seasons[corr_features].corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()

    # Correlation of awards and All-NBA team selection
    awards = ['MVP', 'DPOY', 'ROY', '6MOY', 'MIP', 'All-Star', 'All-Star-MVP', 'POTW', 'POTM']
    for award in awards:
        print(award)
        print(normalized_seasons['All-NBA-Team'].loc[normalized_seasons[award] > 0].value_counts())

    #print(normalized_seasons.describe())

    # Divide the data into training and validation sets
    np.random.seed(42)
    val_seasons = np.random.randint(1988, 2023, 4)
    #print(f'Validation seasons: {val_seasons}')
    X_train = normalized_seasons.loc[~normalized_seasons['SEASON_START'].isin(val_seasons)]
    X_val = normalized_seasons.loc[normalized_seasons['SEASON_START'].isin(val_seasons)]
    y_train_any = X_train['ANY_ALL_NBA']
    y_val_any = X_val['ANY_ALL_NBA']
    y_train = X_train['All-NBA-Team']
    y_val = X_val['All-NBA-Team']

    # Drop unnecessary columns
    columns_to_drop = ['SEASON', 'SEASON_START', 'MATCH_TYPE', 'PLAYER_NAME', 'All-NBA-Team', 'ANY_ALL_NBA',
                        'ANY_ALL_ROOKIE', 'All-Defensive-Team', 'All-Rookie-Team', 'Finals-MVP', 'PLAYER_ID',
                       'ROY', '6MOY', 'MIP']

    X_train = X_train.drop(columns_to_drop, axis=1)
    X_val = X_val.drop(columns_to_drop, axis=1)
    season2324 = normalized_season2324.drop(columns_to_drop, axis=1)

    # Train the Random Forest Classifier model for all All-NBA teams
    RFC_model = RandomForestClassifier(n_estimators=100, random_state=42)
    RFC_model.fit(X_train, y_train)
    RFC_predictions = RFC_model.predict(X_val)
    RFC_confusion_matrix = confusion_matrix(y_val, RFC_predictions)
    RFC_true_positives = np.diag(RFC_confusion_matrix)
    print('Random Forest Classifier ', RFC_confusion_matrix)
    print('Random Forest Classifier ', RFC_true_positives)

    # Train the Random Forest Classifier model for any All-NBA team
    RFC_model_any = RandomForestClassifier(n_estimators=100, random_state=42)
    RFC_model_any.fit(X_train, y_train_any)
    RFC_predictions_any = RFC_model_any.predict(X_val)
    RFC_confusion_matrix_any = confusion_matrix(y_val_any, RFC_predictions_any)
    RFC_true_positives_any = np.diag(RFC_confusion_matrix_any)
    print('Random Forest Classifier Any ', RFC_confusion_matrix_any)
    print('Random Forest Classifier Any ', RFC_true_positives_any)

    # Train the Logistic Regression model for all All-NBA teams
    LR_model = LogisticRegression(random_state=42)
    LR_model.fit(X_train, y_train)
    LR_predictions = LR_model.predict(X_val)
    LR_confusion_matrix = confusion_matrix(y_val, LR_predictions)
    LR_true_positives = np.diag(LR_confusion_matrix)
    print('Logistic Regression ', LR_confusion_matrix)
    print('Logistic Regression ', LR_true_positives)

    # Train the Logistic Regression model for any All-NBA team
    LR_model_any = LogisticRegression(random_state=42)
    LR_model_any.fit(X_train, y_train_any)
    LR_predictions_any = LR_model_any.predict(X_val)
    LR_confusion_matrix_any = confusion_matrix(y_val_any, LR_predictions_any)
    LR_true_positives_any = np.diag(LR_confusion_matrix_any)
    print('Logistic Regression Any ', LR_confusion_matrix_any)
    print('Logistic Regression Any ', LR_true_positives_any)

if __name__ == '__main__':
    all_nba_training()
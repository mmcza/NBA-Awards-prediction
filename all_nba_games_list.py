from nba_api.stats.endpoints import boxscoretraditionalv2
import pandas as pd
import os

# Set pandas options
pd.set_option('display.max_columns', None)

def save_all_games():
    # Load the data
    matches = pd.read_csv("data/matches_46_96.csv")

    match_ids = matches['GAME_ID'].unique()

    all_matches = pd.DataFrame()

    for index, match_id in enumerate(match_ids):
        if len(str(match_id)) < 10:
            match_id_string = "0"*(10-len(str(match_id))) + str(match_id)
        else:
            match_id_string = str(match_id)

        match_result = get_game_result(match_id_string)
        all_matches = pd.concat([all_matches, match_result], ignore_index=True)

        print(f"Processed {index+1} out of {len(match_ids)} matches")

        if index % 100 == 0:
            file_path = os.path.join("data", "all_matches_list.csv")
            # Check if the file already exists
            if os.path.exists(file_path):
                # If the file exists, read the existing data and append the new data
                existing_data = pd.read_csv(file_path)
                combined_data = pd.concat([existing_data, all_matches], ignore_index=True)
                combined_data.to_csv(file_path, index=False)
            else:
                # If the file doesn't exist, save the data directly
                all_matches.to_csv(file_path, index=False)

            all_matches = pd.DataFrame()


    # Save DataFrame to CSV file
    file_path = os.path.join("data", "all_matches_list.csv")
    existing_data = pd.read_csv(file_path)
    combined_data = pd.concat([existing_data, all_matches], ignore_index=True)
    combined_data.to_csv(file_path, index=False)

def get_game_result(game_id):
    # Retrieve box score data for the specified game ID
    boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    boxscore_data = boxscore.get_data_frames()

    # Extract team-level statistics
    team_totals = boxscore_data[1]
    return team_totals

def calculate_not_empty(row, column_to_check, column_to_return):
    if pd.notna(row[column_to_check]):  # Check if FGA is not empty
        return row[column_to_return]
    else:
        return None

def calculate_score_given_player_stats():
    all_matches = pd.read_csv("data/all_matches_stats.csv")

    all_matches['GAME_ID'] = all_matches['GAME_ID'].astype(str)
    all_matches['TEAM_ID'] = all_matches['TEAM_ID'].astype(str)
    all_matches[['minutes', 'seconds']] = all_matches['MIN'].str.split(':', expand=True)
    all_matches['MIN'] = all_matches['minutes'].astype(float) + all_matches['seconds'].astype(float) / 60

    all_matches['FGM_2'] = all_matches.apply(calculate_not_empty, axis=1, args=['FGA', 'FGM'])
    all_matches['FG3M_2'] = all_matches.apply(calculate_not_empty, axis=1, args=['FG3A', 'FG3M'])
    all_matches['FTM_2'] = all_matches.apply(calculate_not_empty, axis=1, args=['FTA', 'FTM'])


    results = all_matches.groupby(['GAME_ID', 'TEAM_ID'])[['PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FGM', 'FGA',
                                                           'FG3M', 'FG3A', 'FTM', 'FTA', 'MIN', 'PF', 'FGM_2', 'FG3M_2', 'FTM_2',
                                                           'OREB', 'DREB']].sum()
    results['FG_PCT'] = results['FGM_2'] / results['FGA']
    results['FG3_PCT'] = results['FG3M_2'] / results['FG3A']
    results['FT_PCT'] = results['FTM_2'] / results['FTA']

    results = results.drop(columns=['FGM_2', 'FG3M_2', 'FTM_2'])

    results = results.reset_index()

    results.to_csv("data/match_results.csv", index=False)

def compare():
    # Load the data
    results = pd.read_csv("data/match_results.csv")
    all_matches = pd.read_csv("data/all_matches_list.csv")

    # iterate over the rows

    different_values = {}

    for index, row in all_matches.iterrows():
        team_id = row['TEAM_ID']
        game_id = row['GAME_ID']
        points_am = row['PTS']
        rebounds_am = row['REB']
        assists_am = row['AST']
        steals_am = row['STL']
        blocks_am = row['BLK']
        turnovers_am = row['TO']

        # get the values from the results dataframe
        points = results.loc[(results['GAME_ID'] == game_id) & (results['TEAM_ID'] == team_id), 'PTS'].values[0]
        rebounds = results.loc[(results['GAME_ID'] == game_id) & (results['TEAM_ID'] == team_id), 'REB'].values[0]
        assists = results.loc[(results['GAME_ID'] == game_id) & (results['TEAM_ID'] == team_id), 'AST'].values[0]
        steals = results.loc[(results['GAME_ID'] == game_id) & (results['TEAM_ID'] == team_id), 'STL'].values[0]
        blocks = results.loc[(results['GAME_ID'] == game_id) & (results['TEAM_ID'] == team_id), 'BLK'].values[0]
        turnovers = results.loc[(results['GAME_ID'] == game_id) & (results['TEAM_ID'] == team_id), 'TO'].values[0]

        # compare the values
        if points != points_am:
            different_values[game_id] = 'PTS'
        if rebounds != rebounds_am:
            different_values[game_id] = 'REB'
        if assists != assists_am:
            different_values[game_id] = 'AST'
        if steals != steals_am:
            different_values[game_id] = 'STL'
        if blocks != blocks_am:
            different_values[game_id] = 'BLK'
        if turnovers != turnovers_am:
            different_values[game_id] = 'TO'

    print(different_values)
    print(len(different_values))


if __name__ == "__main__":
    #save_all_games()
    calculate_score_given_player_stats()
    #compare()

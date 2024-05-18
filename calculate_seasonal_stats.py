import pandas as pd

def prepare_matches_46_96():
    # Load the data
    all_matches_stats = pd.read_csv("data/all_matches_stats_46_96.csv")
    seasons = pd.read_csv("data/NBA_Seasons_Dates.csv")

    # Converting columns to datetime
    date_columns = ['Regular_season_start', 'Regular_season_end', 'Playin_start', 'Playin_end',
                    'Playoffs_start', 'Playoffs_end', 'Finals_start', 'Finals_end']
    seasons[date_columns] = seasons[date_columns].apply(lambda x: pd.to_datetime(x, errors='coerce'))

    # Remove the matches from 1996-97 season because they were available on the NBA website
    season_96_97 = seasons.loc[seasons['Season'] == '1995-96', 'Finals_end']
    all_matches_stats['GAME_DATE'] = pd.to_datetime(all_matches_stats['GAME_DATE'])
    all_matches_stats = all_matches_stats[all_matches_stats['GAME_DATE'] <= season_96_97.values[0]]

    # Drop unnecessary columns and add the season and match type columns
    all_matches_stats.drop(columns=['NICKNAME', 'COMMENT', 'PLUS_MINUS', 'START_POSITION'], inplace=True)
    all_matches_stats['SEASON'] = None
    all_matches_stats['MATCH_TYPE'] = None

    # Get the season for each match and what kind of match was it
    for index, row in seasons.iterrows():
        if row['Season'] == '1946-47':
            all_matches_stats.loc[all_matches_stats['GAME_DATE'] <= row['Finals_end'], 'SEASON'] = row['Season']
        else:
            previous_final = seasons.loc[index - 1, 'Finals_end']
            all_matches_stats.loc[(all_matches_stats['GAME_DATE'] > previous_final) and (all_matches_stats['GAME_DATE'] <= row['Finals_end']), 'SEASON'] = row['Season']

        all_matches_stats.loc[(all_matches_stats['GAME_DATE'] >= row['Regular_season_start']) and (all_matches_stats['GAME_DATE'] <= row['Regular_season_end']), 'MATCH_TYPE'] = "Regular"
        all_matches_stats.loc[(all_matches_stats['GAME_DATE'] >= row['Playoffs_start']) and (all_matches_stats['GAME_DATE'] <= row['Playoffs_end']), 'MATCH_TYPE'] = "Playoffs"
        all_matches_stats.loc[(all_matches_stats['GAME_DATE'] >= row['Finals_start']) and (all_matches_stats['GAME_DATE'] <= row['Finals_end']), 'MATCH_TYPE'] = "Finals"
        if not pd.isna(row['Playin_start']):
            all_matches_stats.loc[(all_matches_stats['GAME_DATE'] >= row['Playin_start']) and (all_matches_stats['GAME_DATE'] <= row['Playin_end']), 'MATCH_TYPE'] = "Play-In"

    # Save the data
    all_matches_stats.to_csv("data/matches_46_96.csv", index=False)

def time_to_float(time_str):
    if ':' in str(time_str):
        minutes, seconds = time_str.split(':')
        return float(minutes) + float(seconds) / 60
    else:
        return float(time_str)

def prepare_matches():
    # Load the data
    all_matches_stats = pd.read_csv("data/all_matches_stats.csv")
    seasons = pd.read_csv("data/NBA_Seasons_Dates.csv")

    # Converting columns to datetime
    date_columns = ['Regular_season_start', 'Regular_season_end', 'Playin_start', 'Playin_end',
                    'Playoffs_start', 'Playoffs_end', 'Finals_start', 'Finals_end']
    seasons[date_columns] = seasons[date_columns].apply(lambda x: pd.to_datetime(x, errors='coerce'))

    all_matches_stats['GAME_DATE'] = pd.to_datetime(all_matches_stats['GAME_DATE'])
    all_matches_stats['MIN'] = all_matches_stats['MIN'].apply(time_to_float)

    # Drop unnecessary columns and add the season and match type columns
    all_matches_stats.drop(columns=['NICKNAME', 'COMMENT', 'PLUS_MINUS', 'START_POSITION'], inplace=True)
    all_matches_stats['SEASON'] = None
    all_matches_stats['MATCH_TYPE'] = None

    # Get the season for each match and what kind of match was it
    for index, row in seasons.iterrows():
        if row['Season'] == '1946-47':
            all_matches_stats.loc[all_matches_stats['GAME_DATE'] <= row['Finals_end'], 'SEASON'] = row['Season']
        else:
            previous_final = seasons.loc[index - 1, 'Finals_end']
            all_matches_stats.loc[(all_matches_stats['GAME_DATE'] > previous_final) & (all_matches_stats['GAME_DATE'] <= row['Finals_end']), 'SEASON'] = row['Season']

        all_matches_stats.loc[(all_matches_stats['GAME_DATE'] >= row['Regular_season_start']) & (all_matches_stats['GAME_DATE'] <= row['Regular_season_end']), 'MATCH_TYPE'] = "Regular"
        all_matches_stats.loc[(all_matches_stats['GAME_DATE'] >= row['Playoffs_start']) & (all_matches_stats['GAME_DATE'] <= row['Playoffs_end']), 'MATCH_TYPE'] = "Playoffs"
        all_matches_stats.loc[(all_matches_stats['GAME_DATE'] >= row['Finals_start']) & (all_matches_stats['GAME_DATE'] <= row['Finals_end']), 'MATCH_TYPE'] = "Finals"
        if not pd.isna(row['Playin_start']):
            all_matches_stats.loc[(all_matches_stats['GAME_DATE'] >= row['Playin_start']) & (all_matches_stats['GAME_DATE'] <= row['Playin_end']), 'MATCH_TYPE'] = "Play-In"

    # Save the data
    all_matches_stats.to_csv("data/matches_player_stats.csv", index=False)

def calculate_not_empty(row, column_to_check, column_to_return):
    if pd.notna(row[column_to_check]):  # Check if FGA is not empty
        return row[column_to_return]
    else:
        return None

def calculate_over_10(row):
    num_of_stats_over_10 = 0
    columns = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    for column in columns:
        if row[column] >= 10:
            num_of_stats_over_10 += 1
    return num_of_stats_over_10

# Calculate the fantasy points for each player (based on NBA glossary)
def calculate_fantasy_points(row):
    fantasy_points = 0
    fantasy_points += row['PTS']
    fantasy_points += row['REB'] * 1.2
    fantasy_points += row['AST'] * 1.5
    fantasy_points += row['STL'] * 3
    fantasy_points += row['BLK'] * 3
    fantasy_points += row['TO'] * -1
    return fantasy_points

# Calculate the Player Impact Estimate (PIE) for each player (based on NBA glossary)
def calculate_pie(row):
    pie = 0
    pie += (row['PTS'] + row['FGM'] + row['FTM'] - row['FGA'] - row['FTA'] + row['DREB'] + 0.5 * row['OREB'] +
            row['AST'] + row['STL'] + 0.5 * row['BLK'] - row['PF'] - row['TO'])

    if (row['PTS_GAME'] + row['FGM_GAME'] + row['FTM_GAME'] - row['FGA_GAME'] - row['FTA_GAME'] +
            row['DREB_GAME'] + 0.5 * row['OREB_GAME'] + row['AST_GAME'] + row['STL_GAME'] + 0.5 * row['BLK_GAME'] -
            row['PF_GAME'] - row['TO_GAME']) == 0:
        pie = 0
    else:
        pie /= (row['PTS_GAME'] + row['FGM_GAME'] + row['FTM_GAME'] - row['FGA_GAME'] - row['FTA_GAME'] +
                row['DREB_GAME'] + 0.5 * row['OREB_GAME'] + row['AST_GAME'] + row['STL_GAME'] + 0.5 * row['BLK_GAME'] -
                row['PF_GAME'] - row['TO_GAME'])
    return pie

def calculate_seasonal_stats():
    # Load the data
    matches = pd.read_csv("data/matches_player_stats.csv")
    team_results = pd.read_csv("data/matches_team_results.csv")

    sum_columns = ['PTS', 'FGM', 'FTM', 'FGA', 'FTA', 'DREB', 'OREB', 'AST', 'STL', 'BLK', 'PF', 'TO']
    grouped_matches = team_results.groupby('GAME_ID')[sum_columns].sum().reset_index()

    # Calculate the team stats
    matches = matches.merge(grouped_matches, on='GAME_ID', suffixes=('', '_GAME'))

    # Calculate winners
    winner_team = team_results.loc[team_results.groupby('GAME_ID')['PTS'].idxmax()]
    winner_team = winner_team[['GAME_ID', 'TEAM_ID']]

    matches['FGM_2'] = matches.apply(calculate_not_empty, axis=1, args=['FGA', 'FGM'])
    matches['FG3M_2'] = matches.apply(calculate_not_empty, axis=1, args=['FG3A', 'FG3M'])
    matches['FTM_2'] = matches.apply(calculate_not_empty, axis=1, args=['FTA', 'FTM'])
    matches['Stats_over_10'] = matches.apply(calculate_over_10, axis=1)
    matches['DD'] = matches['Stats_over_10'].apply(lambda x: 1 if x >= 2 else 0)
    matches['TD'] = matches['Stats_over_10'].apply(lambda x: 1 if x >= 3 else 0)
    matches['FP'] = matches.apply(calculate_fantasy_points, axis=1)
    matches['GP'] = matches['MIN'].notna().astype(int)
    matches['PIE'] =matches.apply(calculate_pie, axis=1)

    # Calculate if team won the match
    matches = matches.merge(winner_team, on='GAME_ID', how='left', suffixes=('', '_WINNER'))
    matches['W'] = (matches['TEAM_ID'] == matches['TEAM_ID_WINNER']).astype(int)
    matches['L'] = (matches['TEAM_ID'] != matches['TEAM_ID_WINNER']).astype(int)
    matches = matches.drop(columns=['TEAM_ID_WINNER'])

    # Calculate the seasonal stats
    seasonal_stats = (matches.groupby(['SEASON', 'MATCH_TYPE', 'PLAYER_NAME',
                                      'PLAYER_ID'])[['GP', 'W', 'L', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FGM',
                                                     'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'FGM_2', 'FG3M_2', 'FTM_2',
                                                     'DD', 'TD', 'FP', 'PIE']].sum().reset_index())

    # Calculate the percentages
    seasonal_stats['FG_PCT'] = round(seasonal_stats['FGM_2'] / seasonal_stats['FGA'], 4)
    seasonal_stats['FG3_PCT'] = round(seasonal_stats['FG3M_2'] / seasonal_stats['FG3A'], 4)
    seasonal_stats['FT_PCT'] = round(seasonal_stats['FTM_2'] / seasonal_stats['FTA'], 4)

    # Save the data
    seasonal_stats.to_csv("data/seasonal_stats.csv", index=False)

if __name__ == "__main__":
    #prepare_matches()
    calculate_seasonal_stats()

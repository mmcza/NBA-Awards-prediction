import pandas as pd

def prepare_matches():
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
            all_matches_stats.loc[(all_matches_stats['GAME_DATE'] > previous_final) & (all_matches_stats['GAME_DATE'] <= row['Finals_end']), 'SEASON'] = row['Season']

        all_matches_stats.loc[(all_matches_stats['GAME_DATE'] >= row['Regular_season_start']) & (all_matches_stats['GAME_DATE'] <= row['Regular_season_end']), 'MATCH_TYPE'] = "Regular"
        all_matches_stats.loc[(all_matches_stats['GAME_DATE'] >= row['Playoffs_start']) & (all_matches_stats['GAME_DATE'] <= row['Playoffs_end']), 'MATCH_TYPE'] = "Playoffs"
        all_matches_stats.loc[(all_matches_stats['GAME_DATE'] >= row['Finals_start']) & (all_matches_stats['GAME_DATE'] <= row['Finals_end']), 'MATCH_TYPE'] = "Finals"
        if not pd.isna(row['Playin_start']):
            all_matches_stats.loc[(all_matches_stats['GAME_DATE'] >= row['Playin_start']) & (all_matches_stats['GAME_DATE'] <= row['Playin_end']), 'MATCH_TYPE'] = "Play-In"

    # Save the data
    all_matches_stats.to_csv("data/matches_46_96.csv", index=False)

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

def calculate_seasonal_stats():
    # Load the data
    matches = pd.read_csv("data/matches_46_96.csv")

    matches['FGM_2'] = matches.apply(calculate_not_empty, axis=1, args=['FGA', 'FGM'])
    matches['FG3M_2'] = matches.apply(calculate_not_empty, axis=1, args=['FG3A', 'FG3M'])
    matches['FTM_2'] = matches.apply(calculate_not_empty, axis=1, args=['FTA', 'FTM'])
    matches['Stats_over_10'] = matches.apply(calculate_over_10, axis=1)
    matches['DD'] = matches['Stats_over_10'].apply(lambda x: 1 if x >= 2 else 0)
    matches['TD'] = matches['Stats_over_10'].apply(lambda x: 1 if x >= 3 else 0)


    # Calculate the seasonal stats
    seasonal_stats = matches.groupby(['SEASON', 'MATCH_TYPE', 'PLAYER_NAME', 'PLAYER_ID'])[['PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA']].sum().reset_index()

    # Calculate the percentages
    seasonal_stats['FG_PCT'] = seasonal_stats['FGM'] / seasonal_stats['FGA']
    seasonal_stats['FG3_PCT'] = seasonal_stats['FG3M'] / seasonal_stats['FG3A']
    seasonal_stats['FT_PCT'] = seasonal_stats['FTM'] / seasonal_stats['FTA']

    # Save the data
    seasonal_stats.to_csv("data/seasonal_stats_46_96.csv", index=False)

if __name__ == "__main__":
    #prepare_matches()
    calculate_seasonal_stats()

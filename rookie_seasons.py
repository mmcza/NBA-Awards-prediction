import pandas as pd
from nba_api.stats.endpoints import playercareerstats

def main():
    # Load the data
    seasonal_stats = pd.read_csv('data/seasonal_stats_with_awards.csv')

    # Extract unique player IDs
    unique_player_ids = seasonal_stats['PLAYER_ID'].unique()
    seasonal_stats['ROOKIE_SEASON'] = False

    for player_id in unique_player_ids:
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        career_data = career_stats.get_data_frames()[0]

        if not career_data.empty:
            # Get the first season (rookie season)
            rookie_season = career_data.iloc[0]['SEASON_ID']

            seasonal_stats.loc[(seasonal_stats['SEASON'] == rookie_season) & (seasonal_stats['PLAYER_ID'] == player_id),
                'ROOKIE_SEASON'] = True

    # save data to csv file
    seasonal_stats.to_csv("data/seasonal_stats_with_awards.csv", index=False)

if __name__ == '__main__':
    main()

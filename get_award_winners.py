from nba_api.stats.endpoints import CommonAllPlayers, PlayerAwards
import pandas as pd
import os

def all_players_from_api():
    # Get all players
    all_players = CommonAllPlayers(is_only_current_season=0, league_id='00')

    # Convert data to DataFrame
    all_players_df = all_players.get_data_frames()[0]

    return all_players_df

def all_players_from_csv():
    # Load the data
    all_players_df = pd.read_csv("data/all_matches_stats.csv")

    # Get unique player IDs
    unique_player_ids = all_players_df[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()

    return unique_player_ids
def main():
    #all_players_df = all_players_from_api()
    all_players_df = all_players_from_csv()

    # Create an empty list to store all player awards DataFrames
    all_player_awards = []

    # Iterate through each player
    for _, player_row in all_players_df.iterrows():
        player_id = player_row['PLAYER_ID']
        player_name = player_row['PLAYER_NAME']

        print(player_name)

        # Get player awards
        player_awards = PlayerAwards(player_id=player_id)

        # Retrieve the data
        player_awards_data = player_awards.get_data_frames()[0]

        # Add player ID and name to the DataFrame
        player_awards_data['PLAYER_ID'] = player_id
        player_awards_data['PLAYER_NAME'] = player_name

        # Append player awards to the all_player_awards list
        all_player_awards.append(player_awards_data)

    # Concatenate all player awards DataFrames into a single DataFrame
    all_player_awards_df = pd.concat(all_player_awards, ignore_index=True)

    # Save the data to a CSV file
    all_player_awards_df.to_csv(os.path.join("data", "nba_player_awards.csv"), index=False)

if __name__ == "__main__":
    main()
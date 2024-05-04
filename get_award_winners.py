from nba_api.stats.endpoints import CommonAllPlayers, PlayerAwards
import pandas as pd
import os

# Get all players
all_players = CommonAllPlayers(is_only_current_season=0, league_id='00')

# Convert data to DataFrame
all_players_df = all_players.get_data_frames()[0]

# Create an empty list to store all player awards DataFrames
all_player_awards = []

# Iterate through each player
for _, player_row in all_players_df.iterrows():
    player_id = player_row['PERSON_ID']
    player_name = player_row['DISPLAY_FIRST_LAST']

    print(player_name)

    # Get player awards
    player_awards = PlayerAwards(player_id=player_id)

    # Retrieve the data
    player_awards_data = player_awards.get_data_frames()[0]

    # Add player ID and name to the DataFrame
    player_awards_data['PERSON_ID'] = player_id
    player_awards_data['PLAYER_NAME'] = player_name

    # Append player awards to the all_player_awards list
    all_player_awards.append(player_awards_data)

# Concatenate all player awards DataFrames into a single DataFrame
all_player_awards_df = pd.concat(all_player_awards, ignore_index=True)

# Save the data to a CSV file
all_player_awards_df.to_csv(os.path.join("data", "nba_players_awards.csv"), index=False)
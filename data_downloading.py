from nba_api.stats.endpoints import PlayerDashboardByYearOverYear, CommonAllPlayers
import pandas as pd

# set pandas options
pd.set_option('display.max_columns', None)

# Get all players
all_players = CommonAllPlayers(is_only_current_season=0, league_id='00')

# Convert data to DataFrame
all_players_df = all_players.get_data_frames()[0]

# Create an empty list to store all player statistics DataFrames
all_player_stats = []

# Get the total number of players
total_players = len(all_players_df)

# Iterate through each player
for idx, (index, player_row) in enumerate(all_players_df.iterrows(), 1):
    player_id = player_row['PERSON_ID']
    player_name = player_row['DISPLAY_FIRST_LAST']

    # Create an instance of the PlayerDashboardByYearOverYear endpoint
    player_dashboard = PlayerDashboardByYearOverYear(player_id=str(player_id), per_mode_detailed='PerGame')

    # Retrieve the data
    player_dashboard_data = player_dashboard.get_data_frames()[1]

    # Add player ID and name to the DataFrame
    player_dashboard_data['PERSON_ID'] = player_id
    player_dashboard_data['PLAYER_NAME'] = player_name

    # Append player statistics to the all_player_stats list
    all_player_stats.append(player_dashboard_data)

    # Print progress
    print(f"Processed player {idx}/{total_players}")

# Concatenate all player statistics DataFrames into a single DataFrame
all_player_stats_df = pd.concat(all_player_stats, ignore_index=True)

# Save the data to a CSV file
all_player_stats_df.to_csv('nba_players_seasonal_stats_per_game.csv', index=False)

# Display the data
print(all_player_stats_df)

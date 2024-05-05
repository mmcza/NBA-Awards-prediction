import pandas as pd
from nba_api.stats.endpoints import ScoreboardV2, BoxScoreTraditionalV2
import datetime
import os
import time

# set pandas options
pd.set_option('display.max_columns', None)

# Define the initial start and end dates
# start_date = datetime.datetime(1947, 10, 1)
# end_date = datetime.datetime(1948, 9, 30)

start_date = datetime.datetime(2023, 10, 1)
end_date = datetime.datetime(2024, 5, 4)

# start timer
start_time = time.time()

# Loop until start_date reaches chosen season
while start_date.year <= 2023:
    # Initialize an empty list to store player stats dataframes
    all_player_stats = []
    print("Currently processing data for the year:", start_date.year)

    # Iterate over each day of the season
    date = start_date
    while date <= end_date:
        print("Currently looking for games @", date)
        # Get the scoreboard for the current date
        scoreboard = ScoreboardV2(game_date=date)

        # Get the game IDs for all games played on the current date
        game_ids = scoreboard.game_header.get_data_frame()['GAME_ID']

        # Iterate over each game ID
        for game_id in game_ids:
            # Get the box score for the current game
            boxscore = BoxScoreTraditionalV2(game_id=game_id)

            # Get the data frame with player stats for the current game
            player_stats = boxscore.player_stats.get_data_frame()

            # Add the date of the game to the DataFrame
            player_stats['GAME_DATE'] = date.strftime('%Y-%m-%d')

            # Append player stats dataframe to the list
            all_player_stats.append(player_stats)

        # Move to the next date
        date += datetime.timedelta(days=1)

    # Concatenate all player stats dataframes into a single dataframe
    full_player_stats_df = pd.concat(all_player_stats, ignore_index=True)

    # Determine the file path
    file_path = os.path.join("data", "all_matches_stats.csv")

    # Check if the file already exists
    if os.path.exists(file_path):
        # If the file exists, read the existing data and append the new data
        existing_data = pd.read_csv(file_path)
        combined_data = pd.concat([existing_data, full_player_stats_df], ignore_index=True)
        combined_data.to_csv(file_path, index=False)
    else:
        # If the file doesn't exist, save the data directly
        full_player_stats_df.to_csv(file_path, index=False)

    # Increment the start_date and end_date by one year
    start_date = start_date.replace(year=start_date.year + 1)
    end_date = end_date.replace(year=end_date.year + 1)

# End timer
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
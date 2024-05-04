import pandas as pd
from nba_api.stats.endpoints import ScoreboardV2, BoxScoreTraditionalV2
import datetime
import os
import time

# set pandas options
pd.set_option('display.max_columns', None)

# Define the initial start and end dates
start_date = datetime.datetime(1947, 10, 1)
end_date = datetime.datetime(1948, 9, 30)

# Function to save game information to CSV file
def save_game_info(game_info):
    file_path = os.path.join("data", "list_of_all_nba_games.csv")
    if os.path.exists(file_path):
        game_info.to_csv(file_path, mode='a', header=False, index=False)
    else:
        game_info.to_csv(file_path, index=False)

# Loop until start_date reaches selected season
while start_date.year <= 2022:
    # Initialize an empty DataFrame to store game information
    game_info_df = pd.DataFrame()

    # Iterate over each day of the season
    date = start_date
    while date <= end_date:
        print("Currently looking for games @", date)
        # Get the scoreboard for the current date
        scoreboard = ScoreboardV2(game_date=date)

        # Get the game IDs for all games played on the current date
        game_ids = scoreboard.game_header.get_data_frame()

        # Append game information to DataFrame
        game_info_df = pd.concat([game_info_df, game_ids])

        # Move to the next date
        date += datetime.timedelta(days=1)

    # Check if the current date is the end of the season (30th September)
    save_game_info(game_info_df)

    # Increment the start_date and end_date by one year
    start_date = start_date.replace(year=start_date.year + 1)
    end_date = end_date.replace(year=end_date.year + 1)

print("All seasons processed and game information saved.")

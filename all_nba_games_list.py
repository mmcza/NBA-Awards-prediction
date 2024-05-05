from nba_api.stats.endpoints import boxscoretraditionalv2
import pandas as pd
import os

# Set pandas options
pd.set_option('display.max_columns', None)

def calculate_seasonal_stats():
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

if __name__ == "__main__":
    calculate_seasonal_stats()

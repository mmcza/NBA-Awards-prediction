import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def all_nba():
    # Load the data
    seasonal_stats = pd.read_csv('data/seasonal_stats_with_awards.csv')
    # Filter the data to only include regular season games
    regular_seasons = seasonal_stats[seasonal_stats['MATCH_TYPE'] == 'Regular']

    def convert_season_to_start_year(season):
        return int(season[:4])

    regular_seasons['SEASON_START'] = regular_seasons['SEASON'].apply(convert_season_to_start_year)
    regular_seasons = regular_seasons.loc[regular_seasons['SEASON_START'] >= 1988]

    # Add a column to determine if player was in any of the All-NBA teams
    regular_seasons['ANY_ALL_NBA'] = regular_seasons['All-NBA-Team'].apply(lambda x: 1 if x > 0 else 0)
    # Add a column to determine if player was in any of the All-Rookie teams
    regular_seasons['ANY_ALL_ROOKIE'] = regular_seasons['All-Rookie-Team'].apply(lambda x: 1 if x > 0 else 0)

    fig, axs = plt.subplots(2, 2)

    # Plot data on each subplot
    axs[0, 0].scatter(regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'PLAYER_NAME'],
                      regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'GP'])
    axs[0, 0].set_title('Games Played by All-NBA Players')

    axs[0, 1].scatter(regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'PLAYER_NAME'],
                      regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'MIN'])
    axs[0, 1].set_title('Minutes Played by All-NBA Players')

    axs[1, 0].scatter(regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'PLAYER_NAME'],
                      regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'PTS'])
    axs[1, 0].set_title('Points Scored by All-NBA Players')

    axs[1, 1].scatter(regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'PLAYER_NAME'],
                      regular_seasons.loc[regular_seasons['ANY_ALL_NBA'] == 1, 'FP'])
    axs[1, 1].set_title('FP by All-NBA Players')

    # Display the plot
    plt.tight_layout()
    plt.show()

    # After plotting, we can see that there can be added some filters to the data
    regular_seasons_len = len(regular_seasons)
    regular_seasons = regular_seasons.loc[regular_seasons['GP'] >= 40]
    regular_seasons = regular_seasons.loc[regular_seasons['MIN'] >= 1250]
    regular_seasons = regular_seasons.loc[regular_seasons['PTS'] >= 333]
    regular_seasons = regular_seasons.loc[regular_seasons['FP'] >= 1250]
    regular_seasons_len_after = len(regular_seasons)
    print(f'Number of rows before filtering: {regular_seasons_len} and after filtering: {regular_seasons_len_after}')

    # Separate the 2023-24 season as it is the season we want to predict
    season2324 = regular_seasons.loc[regular_seasons['SEASON_START'] == 2023]
    season2324 = season2324.loc[season2324['GP'] >= 65]
    regular_seasons = regular_seasons.loc[regular_seasons['SEASON_START'] < 2023]
    print(f'Number of players that possibly can be in All-NBA Team for 2023-24 {len(season2324)}')

if __name__ == '__main__':
    all_nba()
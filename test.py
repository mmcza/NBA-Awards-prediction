import pandas as pd

pd.set_option('display.max_columns', None)

def main():
    seasonal_stats = pd.read_csv('data/seasonal_stats_with_awards.csv')
    seasonal_stats_copy = pd.read_csv('data/seasonal_stats_with_awards_copy.csv')

    # Add information if the player is a rookie
    seasonal_stats['ROOKIE_SEASON'] = False
    for index, row in seasonal_stats_copy.iterrows():
        if row['ROOKIE_SEASON'] == True:
            seasonal_stats.loc[(seasonal_stats['SEASON'] == row['SEASON']) & (seasonal_stats['PLAYER_ID'] == row['PLAYER_ID']),
                'ROOKIE_SEASON'] = True

    # save data to csv file
    seasonal_stats.to_csv("data/seasonal_stats_with_awards.csv", index=False)

if __name__ == '__main__':
    main()

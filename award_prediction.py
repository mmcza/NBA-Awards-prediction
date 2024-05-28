import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
def main():


    weird_stats = pd.read_csv('data/matches_player_stats.csv')

    print(weird_stats[(weird_stats['PLAYER_ID'] == 77182) & (weird_stats['SEASON'] == '1979-80')])
    weird_stats[(weird_stats['PLAYER_ID'] == 77182) & (weird_stats['SEASON'] == '1979-80')].to_csv('data/77182.csv', index=False)
    seasonal_stats = pd.read_csv('data/seasonal_stats_with_awards.csv')
    print(seasonal_stats[seasonal_stats['GP'] > 82])

    #print(seasonal_stats.columns)

if __name__ == '__main__':
    main()
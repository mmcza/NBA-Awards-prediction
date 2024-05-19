import pandas as pd

def main():
    # Load the data
    seasonal_stats = pd.read_csv('data/seasonal_stats_with_awards.csv')


    print(seasonal_stats.columns)

if __name__ == '__main__':
    main()
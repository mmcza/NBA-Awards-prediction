# 2023/2024 NBA Awards prediction
1st 2nd 3rd All-NBA teams and 1st 2nd Rookie All-NBA teams prediction - project for course "Selected topics of machine learning" during 1 semester of Master's

## Table of Contents


## 1. Getting the data

The data was downloaded from [nba.com/stats](https://www.nba.com/stats/) using the [nba_api](https://github.com/swar/nba_api)
library. The data was downloaded for all NBA seasons (1946-47 - 2023-24) and contains:

- player awards,
- player statistics in each season,
- player statistics in each game,
- dates of beginning and end of each season (regular season, playoffs and finals) - based on Wikipedia data.

All data was saved in the `data` directory in the `csv` format.

## 2. Data preprocessing

The data had to be preprocessed because on the website the seasonal statistics were only for seasons 1996-97 - 2023-24.
For the seasons 1946-47 - 1995-96, the data was downloaded for each game and then aggregated to get the seasonal statistics
(if specific statistic was used at the time - [link to list](https://www.nba.com/stats/help/faq#:~:text=HOW%20FAR%20BACK,Goals%3A%201979%2D1980)).
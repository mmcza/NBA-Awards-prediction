# 2023/2024 NBA Awards prediction
1st 2nd 3rd All-NBA teams and 1st 2nd Rookie All-NBA teams prediction - project for course "Selected topics of machine learning"

## Table of Contents

<!-- TOC -->
* [2023/2024 NBA Awards prediction](#20232024-nba-awards-prediction)
  * [Table of Contents](#table-of-contents)
  * [Requirements](#requirements)
  * [1. Getting the data](#1-getting-the-data)
  * [2. Data preprocessing](#2-data-preprocessing)
    * [2.1. Seasons and types of matches](#21-seasons-and-types-of-matches)
    * [2.2. Player statistics](#22-player-statistics)
    * [2.3. Awards](#23-awards)
    * [2.4. Average statistics and normalization](#24-average-statistics-and-normalization)
<!-- TOC -->

## Requirements

The project was written in Python 3.11. The required packages are listed in the [requirements.txt](requirements.txt) file. To install it, run:
```commandline
pip install -r requirements.txt
```

## 1. Getting the data

The data was downloaded from [nba.com/stats](https://www.nba.com/stats/) using the [nba_api](https://github.com/swar/nba_api)
library. The data was downloaded for all NBA seasons (1946-47 - 2023-24) and contains:

- player statistics in each game (downloaded by [this](player_per_match_stats_downloading.py) script) - because the file with statistics from all matches is too big to be uploaded to the repository (around 250MB), it is available in [this Kaggle dataset](https://www.kaggle.com/datasets/marcinmc/nba-players-statistics-seasonal-and-matchbymatch/data),
- team statistics in each game (calculated based on the data from the previous point in [this](nba_matches_team_stats.py) script),
- player statistics in each season (calculated based on the data from the player statistics in [this](calculate_seasonal_stats.py) script),
- player awards (downloaded by [this](get_award_winners.py) script),
- information about rookie seasons of the players (downloaded by [this](rookie_seasons.py) script),
- dates of beginning and end of each season (regular season, playoffs and finals) - based on Wikipedia data.

All data was saved in the `data` directory in the `csv` format.

> [!NOTE]
> I found some mistakes in the data in older seasons - for example, some players were in scoreboard of a match, but they didn't play in that game (they weren't playing for any team from that game). It was usually caused by the same last name of the players and the data was doubled and as a result final scores might differ from the real ones.

## 2. Data preprocessing

The data had to be preprocessed because on the NBA website the seasonal statistics were available only for seasons 1996-97 - 2023-24.
Because of that, the data was downloaded for each game in history of NBA and then aggregated to get the seasonal statistics
(if specific statistic was used at the time - [link to list](https://www.nba.com/stats/help/faq#:~:text=HOW%20FAR%20BACK,Goals%3A%201979%2D1980)).

### 2.1. Seasons and types of matches

Because the All-NBA teams are selected after the regular season, the data was divided into the following types of matches:
- Regular Season,
- All-Star Game,
- Play-in Tournament,
- Playoffs,
- Finals,
- In-Season Tournament Final (other games of the In-Season Tournament are officially considered as Regular Season games).

Inside [NBA_Seasons_Dates.csv](/data/NBA_Seasons_Dates.csv) file there are dates of beginning and end of regular season, playoffs and finals for each season. That information was used to add information about the type of match and season to the statistics.

### 2.2. Player statistics

Apart from the statistics available on the NBA website, the following statistics were calculated:
- `Fantasy Points` - based on the formula: `FP` = `PTS` + 1.2 * `REB` + 1.5 * `AST` + 3 * `STL` + 3 * `BLK` - `TO`,
- `Player Impact Estimate` - based on the formula: `PIE` = (`PTS` + `FGM` + `FTM` - `FGA` - `FTA` + `DREB` + 0.5 * `OREB` + `AST` + `STL` + 0.5 * `BLK` - `PF` - `TO`) / (`GmPTS` + `GmFGM` + `GmFTM` - `GmFGA` - `GmFTA` + `GmDREB` + 0.5 * `GmOREB` + `GmAST` + `GmSTL` + 0.5 * `GmBLK` - `GmPF` - `GmTO`),
- number of statistics in double digits - if the number was >= 2, then the player had a double-double `DD` and if the number was >= 3, then the player had a triple-double `TD`,
- field goals made (and 3PT shots made) only if the number of attempts was available - in older seasons not all of the statistics were saved and that could cause FG% to be over 100%,
- information about win/loss in the match.

After that, the data was summed up to get the seasonal statistics for each player.

### 2.3. Awards

The data about awards was downloaded for each player and information about the following awards were added to the dataset:
- Most Valuable Player,
- Rookie of the Year,
- Defensive Player of the Year,
- Most Improved Player,
- 6th Man of the Year,
- All-NBA teams (1st, 2nd, 3rd),
- All-Defensive teams (1st, 2nd),
- All-Rookie teams (1st, 2nd),
- All-Star Game player,
- All-Star Game MVP,
- Finals MVP,
- number of Player of the Week awards,
- number of Player of the Month awards,
- number of Rookie of the Month awards.

The correlation between the awards and the selection to All-NBA teams since 1988-89 season was checked and that data is shown in the table below (for POTM, POTW it meant that the player won at least one award during the season):

| Award | 1st All-NBA Team | 2nd All-NBA Team | 3rd All-NBA Team | Not selected |
|---------------------|------------------|------|------|--------------|
| MVP                 | 35               | 0    | 0    | 0            |
| DPOY                | 11               | 6    | 8    | 10           |
| ROY                 | 1                | 0    | 1    | 35           |
| 6MOY                | 0                | 0    | 1    | 24           |
| MIP                 | 0                | 5    | 4    | 26           |
| All-Star Game Player | 163              | 156  | 142  | 356          |
| All-Star Game MVP   | 26               | 7    | 1    | 2            |
| POTW                | 151              | 125  | 105  | 423          |
| POTM                | 109              | 56   | 31   | 52           |

The data shows that the MVPs are always selected to 1st All-NBA team, All-Star Game MVPs are usually selected to 1st All-NBA team or 2nd All-NBA team. DPOYs, All-Star Game Players, POTWs and POTMs have high chance to be selected to All-NBA teams.

### 2.4. Average statistics and normalization

The statistics were averaged for each player to get his average impact on the game per match (by doing so the number of games player played doesn't matter). 

Also because basketball and players were evolving over the years, the statistics were normalized so that the player with highest certain statistic would have 1 and the rest of the players would have proportionally lower values.

However this could cause problems with players who played just a few games during a season and had very high statistics in those games. To eliminate the issue, after displaying data for all players who were selected to All-NBA teams, the following filters were applied:
- `Games Played` >= 40,
- `Minutes` played during season >= 1250,
- `Points` scored during season >= 333,
- `Fantasy Points` scored during season >= 1250.

By doing so, the data for seasons 1988-89 till 2023-23 was reduced from 16711 to 6074 players. For season 2023-24 there was also a requirement for `Games Played` >= 65 and that caused that only 146 were eligible for All-NBA teams.
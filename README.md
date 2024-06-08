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
      * [2.3.1. Awards and All-NBA teams](#231-awards-and-all-nba-teams)
    * [2.4. Average statistics and normalization](#24-average-statistics-and-normalization)
    * [2.4.1. Eliminating players with low statistics for All-NBA teams prediction](#241-eliminating-players-with-low-statistics-for-all-nba-teams-prediction)
    * [2.4.2. Eliminating players with low statistics for Rookie All-NBA teams prediction](#242-eliminating-players-with-low-statistics-for-rookie-all-nba-teams-prediction)
    * [2.4.3. Statistics correlation for All-NBA teams prediction](#243-statistics-correlation-for-all-nba-teams-prediction)
  * [3. Splitting the data for training and validation sets](#3-splitting-the-data-for-training-and-validation-sets)
  * [4. Metric](#4-metric)
  * [5. Models](#5-models)
    * [5.1. All-NBA teams prediction](#51-all-nba-teams-prediction)
      * [5.1.1. Baseline model (score: 148.25)](#511-baseline-model-score-14825)
      * [5.1.2. Random Forest Classifier with only per game statistics (score: 141.5)](#512-random-forest-classifier-with-only-per-game-statistics-score-1415)
      * [5.1.3. Random Forest Classifier with prediction voting (score: 154.5)](#513-random-forest-classifier-with-prediction-voting-score-1545)
      * [5.1.4. Comparison of different default models (Logistic Regression, Support Vector Machine, Decision Tree, Random Forest, K-Nearest Neighbors, XGBOOST, LightGBM, Voting Classifier)](#514-comparison-of-different-default-models-logistic-regression-support-vector-machine-decision-tree-random-forest-k-nearest-neighbors-xgboost-lightgbm-voting-classifier)
      * [5.1.5. Hyperparameter tuning and feature selection](#515-hyperparameter-tuning-and-feature-selection)
    * [5.2. Rookie All-NBA teams prediction](#52-rookie-all-nba-teams-prediction)
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

#### 2.3.1. Awards and All-NBA teams

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

Also because basketball and players were evolving over the years, the statistics were normalized so that the player with highest certain statistic in specific season would have value 1 and the rest of the players would have proportionally lower values.

However this could cause problems with players who played just a few games during a season and had very high statistics in those games.

### 2.4.1. Eliminating players with low statistics for All-NBA teams prediction
To eliminate the issue, after displaying data for all players who were selected to All-NBA teams (graph below), the following filters were applied:
- `Games Played` >= 40,
- `Minutes` played during season >= 1250,
- `Points` scored during season >= 333,
- `Fantasy Points` scored during season >= 1250.

![Statistics of players selected to All-NBA teams](/media/1_All_NBA_players_through_years_stats.png)

By doing so, the data for seasons 1988-89 till 2023-23 was reduced from 16711 to 6074 players. For season 2023-24 there was also a requirement for `Games Played` >= 65 and that caused that only 146 were eligible for All-NBA teams.

### 2.4.2. Eliminating players with low statistics for Rookie All-NBA teams prediction

### 2.4.3. Statistics correlation for All-NBA teams prediction

After normalizing the data, the correlation between the normalized statistics and the selection to All-NBA teams was checked. The correlation matrix is shown below:

![Correlation matrix for All-NBA teams prediction](/media/2_All_NBA_players_normalized_stats_correlation.png)

Based on the correlation matrix, the highest importance for the selection to All-NBA teams have the following statistics:
- `Player Impact Estimate`,
- `Fantasy Points`,
- `Points`,
- `Free Throws Made`,
- `Field Goals Made`.

High correlation between those above statistics and being selected to All-NBA teams is understandable as those statistics (apart from Free Throws made) directly show impact on the game. Free Throws Made may be correlated because good players usually play more and create more actions so the possiblity of being fouled is higher.

The least correlated statistics are:
- `Free Throw Percentage`,
- `3PT Field Goals Percentage`,
- `3 PT Field Goals Made`.

The low correlation between 3PT Shot statistics is probably caused by the fact that the Centers and Power Forwards usually don't shoot 3PT shots. And in the past those kind of players also weren't good in Free Throws what explains the low correlation with Free Throw Percentage.

## 3. Splitting the data for training and validation sets

The data was split into training and validation sets and each season is fully in either training or validation set. 4 validation seasons were randomly selected and the score for validation set was calculated as mean value of the metric for each of the 4 seasons.

## 4. Metric

The following metric was used to evaluate the model (proposed by the course lecturer):
 - `+10 points` for each player in correct team,
 - `+8 points` for each player that is classified in a team that's number differ by 1 from the correct one,
 - `+6 points` for each player that is classified in a team that's number differ by 2 from the correct one,
 - `+5 points` if 2 players are in correct team,
 - `+10 points` if 3 players are in correct team,
 - `+20 points` if 4 players are in correct team,
 - `+40 points` if 5 players are in correct team.

That means that the maximum number of points for a season is $5 \cdot (5 \cdot 10 + 40) = 450$.

Using metrics like accuracy would be misleading because the number of players not selected to any of the All-NBA teams is much higher than those who got selected. An example could be to classify every of the 146 players eligible to be selected to All-NBA teams in 2023-24 season as not selected and the accuracy would be 0.89.  

## 5. Models

### 5.1. All-NBA teams prediction

Below are some of the models that were used to predict the players selected to All-NBA teams.

#### 5.1.1. Baseline model (score: 148.25)

The baseline model was a Random Forest Classifier with `n_estimators = 100` that was predicting probability of player being selected to each of the All-NBA teams. The mean score on validation set was 148.25 out of 270 points. The feature importance for the model is shown below:

![Feature importance for baseline model](/media/3_All_NBA_baseline_feature_importance.png)

The baseline model got a high score so it's a good starting point, but also makes it harder to find early improvements.

#### 5.1.2. Random Forest Classifier with only per game statistics (score: 141.5)

After removing the statistics that weren't calculated as mean per game, the score of the model decreased to 141.5.

#### 5.1.3. Random Forest Classifier with prediction voting (score: 154.5)

The model was predicting the probability of player being selected to each of the All-NBA teams and than the predictions were used to calculate voting points from the formula:

$VotPts=5 \cdot P_{1st Team} + 3 \cdot P_{2nd Team} + 1 \cdot P_{3rd Team}$.

The formula is based on the formula to [calculate results of real All-NBA Team voting](https://x.com/NBAPR/status/1793430330113654910/photo/1). After calculating the points, top players were added to each team. The score of the model was 154.5.

Only the mean per game statistics were used as the score was higher than for the model with all statistics.

#### 5.1.4. Comparison of different default models - Logistic Regression, Support Vector Machine, Decision Tree, Random Forest, K-Nearest Neighbors, XGBOOST, LightGBM, Voting Classifier (score: 158.75)

The comparison of the models is shown in the table below:

| Model                  | Only per game<br/> stats + Voting | No per game<br/> stats + Voting | All stats<br/> + Voting | Only per game<br/> stats + No Voting | No per game<br/>stats + No Voting | All stats<br/> + No Voting |
|------------------------|-----------------------------------|---------------------------------|--------------------|--------------------------------------|-----------------------------------|-----------------------|
| Logistic Regression    | 121.75                            | 109.25                          | 109.25             | 116.00                               | 109.25                            | 111.75                |
| Support Vector Machine | 113.25                            | 124.25                          | 124.25             | 118.50                               | 122.75                            | 122.75                |
| Decision Tree          | 120.50                            | 102.75                          | 110.00             | 117.00                               | 86.75                             | 87.75                 |
| Random Forest          | **154.50**                        | **158.75**                      | **145.75**         | **142.00**                           | **149.25**                        | **148.25**            |
| K-Nearest Neighbors    | 106.50                            | 105.25                          | 105.25             | 95.00                                | 104.25                            | 104.25                |
| XGBOOST                | 141.00                            | 143.00                          | 143.25             | 131.75                               | 131.00                            | 138.50                |
| LightGBM               | 135.50                            | 147.75                          | 137.50             | 137.25                               | 140.50                            | 137.50                |
| Voting Classifier*     | 133.75                            | 145.00                          | 136.50             | 139.25                               | 139.50                            | 136.25                |

*Voting Classifier was built from all the above models.

With **bold** are marked the best scores for each configuration.

The best score was achieved by Random Forest Classifier (158.75). Scores above 140 points were achieved also by:
- XGBOOST - in 3 configurations,
- LightGBM - in 2 configurations,
- Voting Classifier - in 1 configuration.

#### 5.1.5. Hyperparameter tuning and feature selection (score 175.75)
Only the 4 models that achieved 140 points at least once were selected for hyperparameter tuning.

The following parameter grid was created (Voting Classifier was built only from other models in this table):

|            | Random Forest Classifier | XGBoost | LightGBM | Voting Classifier                                                                                                                      |
|------------|----------------------|---------|----------|----------------------------------------------------------------------------------------------------------------------------------------|
| Parameters | ```{'n_estimators': [100, 200, 300, 400, 500],```<br/> ```'max_depth': [10, 25, 50, 100, None],```<br/> ```'min_samples_split': [2, 5, 10],```<br/> ```'min_samples_leaf': [1, 2, 4],```<br/>```'max_features': ['sqrt', 'log2']}```  | ```{'n_estimators': [100, 200, 300, 400, 500],```<br/> ```'max_depth': [10, 25, 50, 100, None],```<br/> ```'learning_rate': [0.01, 0.05, 0.1, 0.2],```<br/> ```'subsample': [0.6, 0.8, 1],```<br/> ```'colsample_bytree': [0.5, 0.8, 1],```<br/> ```'gamma': [0, 0.1, 0.2, 0.3, 0.4]}``` | ```{'n_estimators': [100, 200, 300, 400, 500],```<br/> ```'max_depth': [10, 25, 50, 100, None],```<br/> ```'learning_rate': [0.01, 0.05, 0.1, 0.2],```<br/> ```'subsample': [0.6, 0.8, 1],```<br/> ```'colsample_bytree': [0.5, 0.8, 1]}``` | ```{'weights': [[1, 1, 1], [1, 2, 1], [1, 1, 2], [2, 1, 1], [2, 2, 1], [1, 2, 2]]```<br/> ```'voting': 'soft'}``` |

Also the feature selection was implemented. In each iteration there were randomly chosen a random number of features (at least 5) from the list of statistics and for each set of features there were 50 iterations of hyperparameter tuning. 

By randomly choosing the features 50 times and then randomly choosing parameters 50 times, there was a total of 2500 results for each model (10000 in total). Also each model was tested with and without additional voting for the prediction so as a result 20000 combinations were checked. The optimization process took ~6 hours. The best model got score 175.75 (what is a significant improvement over the baseline model). Features and hyperparameters for the best model are as follows:

- model: `Random Forest Classifier`,
- model parameters: `{'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt'}`,
- features: `['STL', 'FTM_2', 'STL_per_GP', 'PIE_per_GP', 'FTA', 'FG3M_per_GP', 'POTM', 'All-Star', 'MIN', 'FGM_2', 'FP', 'DD', 'GP', 'FTA_per_GP', 'PTS', 'REB', 'DPOY', 'FT_PCT', 'REB_per_GP', 'All-Star-MVP', 'L', 'TD', 'FG3_PCT', 'BLK_per_GP', 'PTS_per_GP', 'AST', 'PIE', 'W', 'FG3M_2', 'TO_per_GP', 'FGM_per_GP', 'FGA', 'FTM_per_GP', 'ROTM']`,
- additional voting: `False`.

The best results for each model are shown in the table below:

3 best sets of features (mean value for all models):

#### 5.1.6. How predictions for validation set could be improved

Before 2023/2024 season, each of the All-NBA teams was containing 2 guards, 2 forwards and 1 center (since 2023/24 the voting is positionless). With that in mind the model could be improved by adding information about the position of the player and then filtering the predictions to have correct number of players in each position.

### 5.2. Rookie All-NBA teams prediction

## 6. Predictions for 2023/2024 season

### 6.1. All-NBA teams

Predictions for the 2023/2024 season are based on the best model from the previous section. The predictions are shown in the table below:

| 1st Team | 2nd Team | 3rd Team |
|----------|----------|----------|
| Nikola Jokic | Jalen Brunson | Devin Booker |
| Luka Doncic | Anthony Davis | Domantas Sabonis |
| Shai Gilgeous-Alexander | Anthony Edwards | Damian Lillard |
| Giannis Antetokounmpo | Kevin Durant | Kawhi Leonard |
| Jayson Tatum | LeBron James | Tyrese Haliburton |

### 6.2. Rookie All-NBA teams

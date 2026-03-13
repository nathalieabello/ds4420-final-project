# DS4420 Final Project
#### Nathalie Abello, Akshitha Bhashetty, Eftyghia Kourtelidis

## Introduction
March Madness is the tournament of upsets, where underdogs come out on top more often than in other tournaments. Just last year, over 21.1 million viewers watched the final game [1], and an estimated $3.7 billion was wagered in bets on the men's and women's tournaments combined [2]. This unpredictability definitely engages big crowds, but this also makes it extremely difficult to predict the outcomes of games. We aim to predict not just who will win the game, but by how many points, through three machine learning approaches: a time series model, a multi-layer perceptron, and a Bayesian model. 

## Datasets
1. [Main Dataset](https://www.kaggle.com/datasets/nishaanamin/march-madness-data/data)
    - Information available:
      - Team ratings/rankings
      - Efficiency metrics
      - Coach results
      - Conference results & statistics
      - Public picks
      - Seeds
      - Shooting statistics
      - Tournament Locations
      - Number of upsets per round & which teams upset
      - Etc.

2. [Supplementary Datasets](https://adamcwisports.blogspot.com/p/data.html)
    - Has precomputed metrics like [t-rank](https://adamcwisports.blogspot.com/p/every-possession-counts.html), for example
    - [Available library to get this data nicely](https://github.com/avewright/pybart)
      - Information available:
        - Player stats
        - Team ratings
        - Full schedule with game stats
        - Season-level box-score totals
        - Current teamsheet rankings
        - Etc.
  
3. [Supplementary Dataset](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)
    - Contains both men's and womens NCAA basketball game data across all D1 teams.
      - Information available:
        - Team IDs, Names, and Seeds
        - Final scores of all regular season, conference tournament, and NCAA tournament games since
        - Season-level details including dates and region names
        - Game-by-game stats at a team level (free throws attempted, defensive rebounds, turnovers, etc.)
        - Locations of games
        - Weekly team rankings (men's only) _from Pomeroy, Sagarin, RPI, ESPN, etc._
        - Coaches
        - Conference
        - Alternative team name spellings (potentially use with Google Trends to see how "hot" a team is)
        - Etc.

## Approaches
We will use the difference between the teams' average point differentials as our baseline approach to compare our models against. 

### Time Series
Time series models will fail to capture the nuances of how the opposing team directly affects score differentials, but we are confident in their ability to detect when a team is either a “legacy” or “hot” team, or not, where their past score differentials help predict future ones, rather than the nuances of the matchups themselves. 

### Multi-Layer Perceptron
An MLP can help capture all the nuances that come with the location of a game, opponent, and tons of statistics on both player and team performance. There is no clear-cut way to model this, so using an MLP where the weights adjust as it learns with more data is a good approach to capturing all of these relationships. 

### Bayesian Model
A Bayesian model can be a good fit by treating team performance as uncertain and using some prior knowledge about the team's past performance to update these beliefs as new data is seen. These models are good at handling small amounts of data, which may be beneficial with the short window of limited statistics. 

## References
[1] L. Seitz, “NCAA March Madness Championship Scores 18.1 Million Viewers on CBS, Most-Watched Since 2019,” TheWrap, Apr. 08, 2025. https://www.thewrap.com/ncaa-march-madness-championship-2025-ratings-cbs/

[2] Geoff Zochodne, “March Madness Betting Projected to Hit Record of $4B, Even Without Prediction Markets,” Yahoo Sports, Mar. 09, 2026. https://sports.yahoo.com/articles/march-madness-betting-projected-hit-115900467.html (accessed Mar. 12, 2026).

## Feature Stability Considerations:
- Some features (like pitcher rest days) need special handling for doubleheaders
- Bullpen availability features should account for game 1 usage before game 2
- Team fatigue indicators might need adjustment for back-to-back games
- Can get these features via feature engineering
    - Bullpen availability
        - Avg. amount of relief pitchers/relief innings used in the past 5 games/week

## Relief Pitchers
- Need a way to differentiate starters from relievers
- Easy: treat the starters for each game in the odds table as the only starters, while any other pitching appearance is a reliever
    - Need to account for openers, will explore again
        - Set a starting pitcher innings pitched threshold; If they did not exceed it the threshold, then treat as reliever
            - May complicate/raise inaccuracies when a pitcher is chased out of the first inning

## scrape_fg.py
- When doing .get(<stat>, <fallback>) for counting stats, can just compute the corresponding fallback values from the SQLite tables for the current year

## Other Stat Factors
- Record at that point in the year
- Last 3, 5, 10, 25 game record
- Strength of schedule feature
    - Avg. the total wins over total games played of the opposing team for each team
        - Keep a table of total wins and games played for each team and then fetch when needed
- Platooning
    - Need to keep track of arm splits
        - Need another scraper to get career splits; Also use to it to get batter hitting/pitching side
- BABIP for hitters and pitchers
- Potentially calculate WAR?
    - Batting WAR: 
        - (Batting Runs + Base Running Runs + Fielding Runs + Positional Adjustment + League Adjustment + Replacement Runs) / (Runs Per Win)
            - Batting Runs = wRAA + (lgR/PA – (PF*lgR/PA))*PA + (lgR/PA – (AL or NL non-pitcher wRC/PA))*PA
                - wRAA = ((wOBA – lgwOBA)/wOBA Scale) * PA
            - Base Running Runs
                - wBSR
            - Fielding Runs
                - FG uses UZR, I will use OAA
            - Positional Adjustment
            - League Adjustment
            - Replacement Runs
            - Runs Per Win
- OAA - Has to be monthly, get from Baseball Savant

## Other Non-Stat Factors
- Weather - Done
    - gamePk 632457, datetime 2021-09-16T16:20:00Z -> cancelled game
- Game Time - Done
- Ballpark - Done
- Home/Away - Done
- Are they in Playoff Contention?
    - Easy: Boolean based on team winning%
    - Harder: Keep a standings table
        - Could implement another scraper - Tough
        - Or just compute the standings myself - Most likely option and easier


import pandas as pd
import warnings

matches = pd.read_csv("matches.csv")
warnings.simplefilter(action="ignore", category=FutureWarning)

numeric_cols = [ ## convert obvious numeric columns in one sweep
    "week", "team_score", "opponent_score", "offense_first_downs", "offense_total_yards", "offense_passing_yards",
    "offense_rushing_yards", "offense_turnovers", "defense_first_downs_allowed", "defense_total_yards_allowed",
    "defense_passing_yards_allowed", "defense_rushing_yards_allowed", "defense_turnovers_forced",
    "expected_points_offense", "expected_points_defense", "expected_points_special_teams",
]

matches[numeric_cols] = matches[numeric_cols].apply(pd.to_numeric, errors="coerce") ## convert to numerics

dt_str = (matches["date"].astype(str).str.strip() + " " + matches["season"].astype(str) + " " + matches["time"].astype(str).str.replace("ET", "", regex=False).str.strip())
matches["kickoff_at"] = pd.to_datetime(dt_str, format="%B %d %Y %I:%M%p", errors="coerce")
matches["home_away_code"] = matches["home_away"].astype("category").cat.codes ## converting home/away values
matches["opp_code"] = matches["opponent"].astype("category").cat.codes ## converting opp values
matches["day_code"] = matches["kickoff_at"].dt.dayofweek ## convert days to ints ex. mon = 0, thu = 3, sun = 6
matches["hour"] = matches["kickoff_at"].dt.hour ## get hour of matches as an int

matches["target"] = (matches["result"] == "W").astype("int") ## the target to predict whether the team will win or not ex. W = 1, L = 0

print(matches.head())
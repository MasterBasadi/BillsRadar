import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

map_values = { ## normalizing names to maintain name consistency
    "BUF": "Buffalo Bills", "MIA": "Miami Dolphins", "NYJ": "New York Jets", "NWE": "New England Patriots",
    "RAV": "Baltimore Ravens", "PIT": "Pittsburgh Steelers", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "HTX": "Houston Texans", "CLT": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "OTI": "Tennessee Titans",
    "KAN": "Kansas City Chiefs", "SDG": "Los Angeles Chargers", "DEN": "Denver Broncos", "RAI": "Las Vegas Raiders",
    "PHI": "Philadelphia Eagles", "DAL": "Dallas Cowboys", "NYG": "New York Giants", "WAS": "Washington Commanders",
    "DET": "Detroit Lions", "MIN": "Minnesota Vikings", "GNB": "Green Bay Packers", "CHI": "Chicago Bears",
    "TAM": "Tampa Bay Buccaneers", "ATL": "Atlanta Falcons", "CAR": "Carolina Panthers", "NOR": "New Orleans Saints",
    "LAR": "Los Angeles Rams", "SEA": "Seattle Seahawks", "CRD": "Arizona Cardinals", "SFO": "San Francisco 49ers"
}
numeric_cols = [ ## convert obvious numeric columns
    "week", "team_score", "opponent_score", "offense_first_downs", "offense_total_yards", "offense_passing_yards",
    "offense_rushing_yards", "offense_turnovers", "defense_first_downs_allowed", "defense_total_yards_allowed",
    "defense_passing_yards_allowed", "defense_rushing_yards_allowed", "defense_turnovers_forced",
    "expected_points_offense", "expected_points_defense", "expected_points_special_teams", "season"
]
cols = [ ## stats to take rolling avg for
    "team_score","opponent_score","offense_first_downs","offense_total_yards", "offense_passing_yards",
    "offense_rushing_yards","offense_turnovers","defense_first_downs_allowed","defense_total_yards_allowed",
    "defense_passing_yards_allowed","defense_rushing_yards_allowed","defense_turnovers_forced",
    "expected_points_offense","expected_points_defense","expected_points_special_teams"
]

class MissingDict(dict): ## renaming abbreviations to full team names ex. Buff -> Buffalo Bills
    __missing__ = lambda self, key: key

def rolling_average(group, cols, new_cols):
    grp = group.sort_values("kickoff_at")
    rolling_stats = grp[cols].rolling(3, closed='left').mean()  # ## take data from past three weeks while excluding current week
    grp[new_cols] = rolling_stats.values
    grp = grp.dropna(subset=new_cols)  # ## removes previous n/a data rows aka games before week 1
    return grp

def make_predictions(data, predictors):
    train = data[data["kickoff_at"] < "2024-09-07"] ## training model with all data preceding 2025 week 4
    test  = data[data["kickoff_at"] > "2024-09-08"] ## training model with all data proceeding with 2025 week 4
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors]) ## making prediction
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
    precision = precision_score(test["target"], preds) ## testing precision
    acc = accuracy_score(test["target"], preds) ## testing accuracy
    return combined, precision, acc

def main():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    matches = pd.read_csv("2021_2024_matches.csv")

    matches[numeric_cols] = matches[numeric_cols].apply(pd.to_numeric, errors="coerce") # numeric coercions

    dt_str = (matches["date"].astype(str).str.strip() + " " + matches["season"].astype(str) + " " + matches["time"].astype(str).str.replace("ET", "", regex=False).str.strip()) ## build kickoff datetime
    matches["kickoff_at"] = pd.to_datetime(dt_str, format="%B %d %Y %I:%M%p", errors="coerce")

    matches["home_away_code"] = matches["home_away"].astype("category").cat.codes ## converting home/away values
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes ## converting opp values
    matches["day_code"] = matches["kickoff_at"].dt.dayofweek ## convert days to ints ex. mon = 0, thu = 3, sun = 6
    matches["hour"] = matches["kickoff_at"].dt.hour ## get hour of matches as an int
    matches["target"] = (matches["result"] == "W").astype("int") ## the target to predict whether the team will win or not ex. w = 1, l = 0

    new_cols = [f"{c}_rolling" for c in cols]
    matches_rolling = (matches.groupby("team").apply(lambda t: rolling_average(t, cols, new_cols)).droplevel("team").reset_index(drop=True))

    base_predictors = ["home_away_code", "opp_code", "day_code", "hour"] ## predictors
    predictors = base_predictors + new_cols

    combined, precision, acc = make_predictions(matches_rolling, predictors) ## fit/predict
    print("Precision:", precision)
    print("Accuracy:", acc)

    mapping = MissingDict(**map_values) ## map team codes to full names and merge for convenience
    out = combined.merge(matches_rolling[["kickoff_at", "team", "opponent", "result"]],left_index=True, right_index=True) ## combining both team and opponent dataframes into one
    out["new team"] = out["team"].map(mapping)
    merged = out.merge(out, left_on=["kickoff_at", "new team"], right_on=["kickoff_at", "opponent"])

if __name__ == "__main__":
    main()
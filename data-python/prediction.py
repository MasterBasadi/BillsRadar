from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pandas as pd
import warnings
import os

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
    rolling_stats = grp[cols].rolling(3, closed='left').mean() ## take data from past three weeks while excluding current week
    grp[new_cols] = rolling_stats.values
    grp = grp.dropna(subset=new_cols) ## removes previous n/a data rows aka games before week 1
    return grp

def make_predictions(data: pd.DataFrame, predictors, targets):
    train = data[data["kickoff_at"] < "2024-09-07"]  ## training cutoff
    test  = data[data["kickoff_at"] > "2024-09-08"]  ## testing window
    train = train.dropna(subset=predictors + targets) ## drop rows with missing inputs/targets
    test  = test.dropna(subset=predictors + targets)
    ## xgboost regressor model fitted with basic params
    base = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=1, n_jobs=-1)
    model = MultiOutputRegressor(base)
    model.fit(train[predictors], train[targets])

    preds = model.predict(test[predictors])
    preds_df = pd.DataFrame(preds, columns=[f"pred_{t}" for t in targets], index=test.index)

    mae = {t: float(mean_absolute_error(test[t], preds_df[f"pred_{t}"])) for t in targets} ## evaluate mae and R^2 per target
    r2  = {t: float(r2_score(test[t], preds_df[f"pred_{t}"])) for t in targets}

    preds_df["pred_win"] = (preds_df["pred_team_score"] > preds_df["pred_opponent_score"]).astype(int) ## derive win/loss from predicted scores
    actual_win = (test["team_score"] > test["opponent_score"]).astype(int)
    win_acc = float((preds_df["pred_win"] == actual_win).mean())

    # attach columns from test for inspection
    out = pd.concat([test[["kickoff_at","team","opponent"] + targets].reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)
    return out, mae, r2, win_acc

def main():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    matches = pd.read_csv("data/2021_2024_matches.csv")

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

    out, mae, r2, acc = make_predictions(matches_rolling, predictors, cols) ## fit/predict
    print("MAE per target:", mae)
    print("R2 per target :", r2)
    print("Derived win accuracy from score preds:", round(acc, 3))

    mapping = MissingDict(**map_values) ## map team codes to full names and merge for convenience
    out = out.merge(matches_rolling[["kickoff_at", "team", "opponent", "result"]], on=["kickoff_at", "team", "opponent"], how="left") ## combining both team and opponent dataframes into one
    out["new team"] = out["team"].map(mapping)
    merged = out.merge(out, left_on=["kickoff_at", "team"], right_on=["kickoff_at", "opponent"], suffixes=("_team", "_opp"))

    os.makedirs("out", exist_ok=True)
    out.to_csv("out/team_predictions.csv", index=False) ## export to csv without index column
    merged.to_csv("out/h2h_predictions.csv", index=False)

if __name__ == "__main__":
    main()
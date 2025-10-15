from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import warnings
import os

map_values = { ## normalizing names to maintain name consistency
    "BUF":"Buffalo Bills","MIA":"Miami Dolphins","NYJ":"New York Jets","NWE":"New England Patriots",
    "RAV":"Baltimore Ravens","PIT":"Pittsburgh Steelers","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
    "HTX":"Houston Texans","CLT":"Indianapolis Colts","JAX":"Jacksonville Jaguars","OTI":"Tennessee Titans",
    "KAN":"Kansas City Chiefs","SDG":"Los Angeles Chargers","DEN":"Denver Broncos","RAI":"Las Vegas Raiders",
    "PHI":"Philadelphia Eagles","DAL":"Dallas Cowboys","NYG":"New York Giants","WAS":"Washington Commanders",
    "DET":"Detroit Lions","MIN":"Minnesota Vikings","GNB":"Green Bay Packers","CHI":"Chicago Bears",
    "TAM":"Tampa Bay Buccaneers","ATL":"Atlanta Falcons","CAR":"Carolina Panthers","NOR":"New Orleans Saints",
    "LAR":"Los Angeles Rams","SEA":"Seattle Seahawks","CRD":"Arizona Cardinals","SFO":"San Francisco 49ers"
}

numeric_cols = [ ## convert obvious numeric columns
    "week","team_score","opponent_score","offense_first_downs","offense_total_yards","offense_passing_yards",
    "offense_rushing_yards","offense_turnovers","defense_first_downs_allowed","defense_total_yards_allowed",
    "defense_passing_yards_allowed","defense_rushing_yards_allowed","defense_turnovers_forced",
    "expected_points_offense","expected_points_defense","expected_points_special_teams","season"
]

cols = [ ## stats to take rolling avg for as well as pd fprmatting
    "team_score","opponent_score","offense_first_downs","offense_total_yards","offense_passing_yards",
    "offense_rushing_yards","offense_turnovers","defense_first_downs_allowed","defense_total_yards_allowed",
    "defense_passing_yards_allowed","defense_rushing_yards_allowed","defense_turnovers_forced",
    "expected_points_offense","expected_points_defense","expected_points_special_teams"
]

class MissingDict(dict): ## renaming abbreviations to full team names ex. Buff -> Buffalo Bills
    __missing__ = lambda self, key: key

def rolling_average(group, cols, new_cols, league_means=None):
    grp = group.sort_values("kickoff_at").copy()
    past3 = grp[cols].rolling(3, min_periods=1).mean().shift(1) ## prior-only 3-game mean
    team_prior = grp[cols].expanding(min_periods=1).mean().shift(1) ## prior-only team expanding mean
    tmp = past3.fillna(team_prior)
    if league_means is not None:
        tmp = tmp.fillna(value=league_means) ## league_means should be a series indexed by col names
    grp[new_cols] = tmp.values
    return grp


def make_predictions(data: pd.DataFrame, predictors, targets):
    train = data[data["kickoff_at"] < "2024-09-07"].copy() ## training cutoff
    test  = data[data["kickoff_at"] > "2024-09-08"].copy() ## testing cutoff
    for df in (train, test):
        df[predictors] = df[predictors].replace([np.inf,-np.inf], np.nan) ## fix bad data
    train = train.dropna(subset=predictors)
    y_ok = train[targets].notna().all(axis=1)
    if (~y_ok).sum(): print(f"[make_predictions] Dropping {(~y_ok).sum()} train rows with NaN targets.")
    train = train.loc[y_ok]
    test  = test.dropna(subset=predictors)
    ## xgboost regressor model fitted with basic params
    base = XGBRegressor(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=1,n_jobs=-1)
    model = MultiOutputRegressor(base).fit(train[predictors], train[targets])

    preds = model.predict(test[predictors])
    preds_df = pd.DataFrame(preds, columns=[f"pred_{t}" for t in targets], index=test.index)

    eval_mask = test[targets].notna().all(axis=1)
    mae = {t: float(mean_absolute_error(test.loc[eval_mask, t], preds_df.loc[eval_mask, f"pred_{t}"])) if eval_mask.any() else float("nan") for t in targets} ## evaluate mae and r^2 per target
    r2 = {t: float(r2_score(test.loc[eval_mask, t], preds_df.loc[eval_mask, f"pred_{t}"])) if eval_mask.any() else float("nan") for t in targets}

    preds_df["pred_win"] = (preds_df["pred_team_score"] > preds_df["pred_opponent_score"]).astype(int) ## attach columns from test for inspection
    out = pd.concat([ test[["boxscore_link","season","week","kickoff_at","team","opponent","home_away","day_code"] + targets].reset_index(drop=True), preds_df.reset_index(drop=True) ], axis=1)

    actual_win = (out["team_score"] > out["opponent_score"]).astype(int)
    win_acc = float((out["pred_win"] == actual_win).mean()) if len(out) else float("nan")
    return out, mae, r2, win_acc

def main():
    warnings.simplefilter("ignore", category=FutureWarning)
    matches = pd.read_csv("data/2021_2024_matches.csv")
    matches[numeric_cols] = matches[numeric_cols].apply(pd.to_numeric, errors="coerce") ## convert to numeric values

    dt_str = (matches["date"].astype(str).str.strip() + " " + matches["season"].astype(str) + " " + matches["time"].astype(str).str.replace("ET","",regex=False).str.strip())
    matches["kickoff_at"] = pd.to_datetime(dt_str, format="%B %d %Y %I:%M%p", errors="coerce") ## convert to a datetime

    matches["home_away_code"] = matches["home_away"].astype("category").cat.codes ## converting home/away values
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes ## converting opp values
    matches["day_code"] = matches["kickoff_at"].dt.dayofweek ## convert days to ints ex. mon = 0, thu = 3, sun = 6
    matches["hour"] = matches["kickoff_at"].dt.hour ## get hour of matches as an int
    matches["target"] = (matches["result"] == "W").astype("int") ## the target to predict whether the team will win or not ex. w = 1, l = 0

    league_means = matches[cols].mean(numeric_only=True).fillna(0.0) ## avg data used for predictions without proper data
    new_cols = [f"{c}_rolling" for c in cols]

    matches_rolling = (matches.groupby("team", group_keys=False).apply(lambda t: rolling_average(t, cols, new_cols, league_means=league_means)).reset_index(drop=True))

    base_predictors = ["home_away_code","opp_code","day_code","hour"] ## predictors
    predictors = base_predictors + new_cols

    out, mae, r2, acc = make_predictions(matches_rolling, predictors, cols) ## fit/predict
    print("MAE per target:", mae)
    print("R2 per target :", r2)
    print("Derived win accuracy from score preds:", round(acc, 2))

    mapping = MissingDict(**map_values) ## map team codes to full names and merge for convenience
    out["team_full"] = out["team"].map(mapping) ## map if it's still an abbreviation
    out["opponent_full"] = out["opponent"].map(mapping)
    out["week"] = pd.to_numeric(out["week"], errors="coerce") ## sort weeks numerically
    out["day"] = out["kickoff_at"].dt.day_name().str[:3]
    out["pred_result"] = (out["pred_team_score"] > out["pred_opponent_score"]).map({True:"W", False:"L"})

    pred_cols = [c for c in out.columns if c.startswith("pred_")]

    final_cols = ["season","week","day","kickoff_at","pred_result","team_full","opponent_full"] + cols + pred_cols
    team_predictions = (out[final_cols].rename(columns={"team_full":"team","opponent_full":"opponent"}).sort_values(["team","season","week","kickoff_at"]).reset_index(drop=True))

    os.makedirs("out", exist_ok=True)
    team_predictions.to_csv("out/team_predictions.csv", index=False) ## export to csv without index column
    print("Rows exported:", len(team_predictions))

if __name__ == "__main__":
    main()

## Inspired by DataQuest Tutorial
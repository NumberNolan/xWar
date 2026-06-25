# xWAR — Expected WAR for Position Players (2021–2025)
 
An Expected WAR (xWAR) metric for MLB position players, built from a merged 2021–2025 Statcast
dataset. xWAR re-derives WAR from the underlying batted-ball, baserunning, and fielding inputs that
*produce* performance, rather than performance itself — the idea being to estimate what a player's
WAR "should" look like given their skill profile.
 
## What this is
 
xWAR is built from four components, mirroring how public WAR is constructed:
 
| Component | Inputs | Notes |
|---|---|---|
| **Hitting** | BB%, K%, Barrel% + batted-ball profile (EV, HardHit%, EV90, MaxEV, LA, LD%, GB%, FB%, Pull%) | Predicts xwOBA, converted to batting runs above league average |
| **Baserunning** | Sprint Speed, xwOBA, SB/CS/SB% | Predicts BsR; including real stolen-base attempts (not just raw speed) substantially improved fit over speed alone |
| **Fielding** | FRM (catchers, already in runs) / OAA → runs (everyone else) | OAA converted using Statcast's published conversion: 0.90 runs/out (OF), 0.75 runs/out (IF) |
| **Positional adjustment + replacement level** | Standard runs-per-600-PA tables, scaled by playing time | Anchors the metric to replacement level, not league average |
 
Runs above replacement (RAR) is converted to wins using `RUNS_PER_WIN`, and batting runs use
`WOBA_SCALE` to convert wOBA points to runs. Rather than hardcoding the standard FanGraphs Guts
constants, **both constants are calibrated empirically** by solving for the values that minimize error
against actual WAR — see [Calibration](#calibration-of-woba_scale-and-runs_per_win) below.
 
## Repo structure
 
```
.
├── xWAR_model.ipynb   # full pipeline: clean → model → calibrate → validate → export
├── expectedwar.csv                  # input dataset
└── xwar_all_player_seasons.csv   # output: xWAR for every player-season, all three models
```
 
## Data
 
The notebook expects a CSV (`expectedwar.csv`) with one row per player-season-position, merged from
Statcast and FanGraphs leaderboard exports (2021–2025). Expected columns include percent-formatted
rate stats (`bb`, `k`, `barrel`, `hardhit`, etc.), batted-ball profile fields (`ev`, `maxev`, `la`,
`ld`, `gb`, `fb`, `pull`), baserunning fields (`sprint_speed`, `sb`, `cs`, `sb%`), fielding (`oaa`,
`frm`), and the target columns (`woba`, `bsr`, `war`).
 
## Models
 
Three models are fit and compared for both the hitting and baserunning components:
 
- **Linear Regression**
- **Random Forest**
- **XGBoost**
Each is evaluated on an identical held-out test split (80/20, fixed random seed), so the comparison
table at the end of the notebook is apples-to-apples across all three.
 
### Model selection: Linear Regression
 
**Linear Regression is the model used for the final xWAR metric.** Across testing, it produced the
best (or statistically tied for best) held-out R² on both the hitting and baserunning components — the
more flexible tree-based models (Random Forest, XGBoost) did not produce a meaningful improvement
despite their added complexity. Given that Linear Regression matches or beats the alternatives on
accuracy, it's the clear choice on simplicity grounds too:
 
- **Fully interpretable** — each coefficient reads directly as "runs (or wOBA points) of value per
  unit of the underlying stat," which is useful both for sanity-checking the model and for explaining
  *why* a player's xWAR is what it is.
- **No hyperparameter tuning** — Random Forest and XGBoost both require tuning (tree depth, learning
  rate, number of estimators, subsampling) to avoid overfitting on a dataset of this size; Linear
  Regression has no such surface to tune or get wrong.
- **More stable across re-runs and seasons** — tree ensembles are more prone to fold-to-fold variance
  on a relatively small number of features and player-seasons; the linear fit is far less sensitive to
  the train/test split.
- **Easier to maintain and extend** — adding a new feature, refitting on an updated season, or
  recalibrating constants is just re-running a closed-form fit, not re-tuning a model.
In short: when the simpler model isn't meaningfully worse on the metric that matters, the added
complexity of Random Forest/XGBoost isn't earning its keep. The notebook keeps all three models in
place for transparency and so the comparison can be re-checked as the dataset grows — but Linear
Regression is the one whose output (`xWAR_lin`) should be treated as the headline metric.

 
## Calibration of `WOBA_SCALE` and `RUNS_PER_WIN`
 
Rather than using the standard FanGraphs Guts placeholder constants, the notebook solves for the
`WOBA_SCALE` and `RUNS_PER_WIN` values that minimize squared error against actual WAR. This works
because, once xwOBA and xBsR are already predicted, the rest of the pipeline is linear in exactly two
unknowns:
 
```
xWAR = c1 · raw_woba_diff_term + c2 · other_runs
  c1 = 1 / (WOBA_SCALE · RUNS_PER_WIN)
  c2 = 1 / RUNS_PER_WIN
```
 
A no-intercept OLS fit on the training set recovers `c1`/`c2` exactly, which are then converted back
into `WOBA_SCALE` and `RUNS_PER_WIN`. This is done **separately for each model's pipeline** (lin/rf/xgb
each get their own constants), since each model's xwOBA/xBsR scale differs slightly.
 
## Running it
 
```bash
pip install pandas numpy scikit-learn xgboost matplotlib
jupyter notebook xWAR_model_v4_calibrated.ipynb
```
The notebook will export `xwar_all_player_seasons.csv` with xWAR from all three models for every player-season in the
dataset, alongside the model comparison table and validation plots used to support the model choice
above.
 
## Limitations / next steps
 
- The calibrated `WOBA_SCALE` / `RUNS_PER_WIN` reflect the constants that best map *this specific
  model's* RAR onto actual WAR — not an independent re-derivation of the "true" FanGraphs constants.
  Any systematic bias in the positional/fielding constants gets partially absorbed into
  `RUNS_PER_WIN` rather than isolated.
- Worth checking the two calibration features (`raw_woba_diff_term`, `other_runs`) for collinearity
  before trusting the calibrated constants at face value.
- `bolts`, `hp_to_1b`, and `competitive_runs` are loaded but unused in the final baserunning model —
  they showed mixed/marginal value in testing once SB/CS/SB% were included, but are easy to re-test.

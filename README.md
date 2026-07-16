# xWAR: Expected Wins Above Replacement

A from-scratch WAR estimation framework built on Statcast/FanGraphs data (2021–2025), covering both **position players** and **pitchers**. The goal is to estimate a player's WAR from underlying skill indicators (batted-ball quality, plate discipline, sprint speed, etc.) rather than from realized outcomes — separating true talent from luck/sequencing noise.

Both pipelines use **linear regression** as the final model. Random forest and XGBoost were evaluated as alternatives and underperformed linear regression on held-out R², so linear regression was kept as the production model for both position players and pitchers.

## Repo contents

| File | Description |
|---|---|
| `xWAR_hitters.ipynb` | Trains and validates the position-player pipeline on 2021–2025 data. Exports `xwar_hitters.csv`. |
| `xWAR_pitchers.ipynb` | Trains and validates the pitcher pipeline on 2021–2025 data. Exports `xwar_pitchers.csv`. |
| `xWAR_test_new_data.ipynb` | Scores any new hitter or pitcher dataset against the finalized linear models — pure inference, no fitting on the new data. |
| `xwar_hitters.csv` | Output of `xWAR_hitters.ipynb` — one row per player-season (2021–2025), with `xWAR_lin`, `xWAR_rf`, `xWAR_xgb`. |
| `xwar_pitchers.csv` | Output of `xWAR_pitchers.ipynb` — one row per pitcher-season (2021–2025), with `xWAR_lin`, `xWAR_rf`, `xWAR_xgb`, plus `xra9` and `fip` for reference. |

---

## Position Players

### Overview
xWAR for hitters is built from four components, each modeled or calculated independently, then combined into a single runs-above-replacement figure and converted to wins:

```
xWAR = (Batting Runs + Baserunning Runs + Fielding Runs + Positional Adjustment + Replacement Runs) / RUNS_PER_WIN
```

### 1. Hitting model (xwOBA)
A linear regression predicts **wOBA** from batted-ball and plate-discipline inputs:

- `bb_pct`, `k_pct` — plate discipline
- `barrel_pct`, `hardhit_pct`, `ev90` (90th percentile exit velo) — batted-ball quality
- `la` (launch angle), `ld_pct`, `fb_pct` — batted-ball shape

Expected stats (xwOBA, xBA, etc.) were deliberately **excluded** from the feature set to avoid circularity — the model predicts wOBA from the same underlying process that generates expected stats, rather than using expected stats as an input to predict themselves.

`ev` (average exit velocity), `maxev` (max exit velocity), and `pull_pct` were tested and removed. A p-value/VIF review found `maxev` (p=0.297) and `pull_pct` (p=0.164) were not statistically significant predictors of wOBA, and `ev`/`maxev`/`ev90` had severe multicollinearity (VIF in the thousands) as three overlapping measurements of the same exit-velocity distribution. Dropping all three cost under 0.002 held-out R².

### 2. Baserunning model (xBsR)
A separate linear regression predicts **BsR** (baserunning runs) from:

- `sprint_speed`
- `xwoba` (output of the hitting model — baserunning value is partly a function of how often a player is on base)
- `sb`, `cs` — stolen base activity

`sb_pct` was tested and removed — it's a direct function of `sb` and `cs`, both already in the model as raw counts, and added no independent signal once they were included.

### 3. Fielding runs
Not modeled — calculated directly from measured defensive metrics:
- **Catchers**: FRM (framing runs) used directly.
- **All other positions**: OAA (Outs Above Average) converted to runs via a position-specific factor (infield ≈ 0.75 runs/OAA, outfield ≈ 0.90 runs/OAA).

### 4. Positional adjustment & replacement level
Standard sabermetric constants, pro-rated to playing time (PA / 600):
- Positional adjustment ranges from +9.5 runs/600 PA (C) to -15 runs/600 PA (DH).
- Replacement level is a flat +20 runs/600 PA added to every player, representing the gap between replacement-level and average performance.

### 5. Calibration
`WOBA_SCALE` (converts wOBA above league average into runs) and `RUNS_PER_WIN` (converts total runs above replacement into wins) are **not** assumed from published constants — they're empirically fit via a no-intercept linear regression of actual WAR against the raw wOBA-differential term and the other-runs total, using 2021–2025 data. This lets the model calibrate itself to the specific run-scoring environment of the training window rather than relying on external published constants.

### Multi-position handling
Players who logged time at multiple positions in a season are collapsed into a single row (grouped by player/season), with OAA summed across positions and other fields taken as the primary-position value, rather than treating each position as a separate row.

### Model comparison
Linear regression outperformed both Random Forest and XGBoost on held-out R² for the hitting model, and was selected as the final approach.

---

## Pitchers

### Overview
xWAR for pitchers uses an **xRA9 approach**: predict a runs-allowed rate from pitch-level skill indicators, scale by innings pitched, and compare to a replacement-level baseline.

```
xWAR = (RA9_replacement − xRA9) × (IP / 9) / RUNS_PER_WIN
```

### 1. xRA9 model
A linear regression predicts **ERA** (used as the runs-allowed proxy) from six features:

- `k_pct`, `bb_pct` — strikeout and walk rate
- `hr_fb_pct` — home run per fly ball rate
- `gb_pct`, `fb_pct` — batted-ball profile
- `hardhit_pct` — contact quality allowed

These are close analogues to FIP's own components (K, BB, HR, contact quality), which is intentional — the model is designed to isolate the parts of run prevention that are pitcher skill rather than defense, sequencing, or luck on balls in play. In validation, xRA9 correlates much more strongly with FIP than with ERA, confirming it's capturing the skill-based signal rather than the noisier, defense-dependent ERA outcome directly.

A p-value/VIF review of these six features (matching the review done on the hitting model) found **no features worth removing** — every coefficient is significant at p < 0.001.

### 2. Calibration
`RA9_replacement` (the runs-per-9 threshold that defines replacement level) and `RUNS_PER_WIN` are empirically fit via a no-intercept linear regression of actual WAR against `(IP/9)` and `xRA` (xRA9 × IP/9), using 2021–2025 data — the same self-calibrating approach used on the hitter side, rather than relying on externally published constants.

---

## Testing on new data

`xWAR_test_new_data.ipynb` is a standalone notebook for scoring any new hitter or pitcher dataset — a new season, a subset of players, etc. — against the finalized models, without editing the training notebooks. It re-trains fresh on the 2021–2025 data (identical feature sets and cleaning logic to `xWAR_hitters.ipynb` / `xWAR_pitchers.ipynb`) and then scores the new file via `.predict()` only; the new data never touches `.fit()`. Set the input/output file names in the config cell at the top and run all cells.

---

## Shared design principles

- **No circularity**: expected/estimated stats are never used as inputs to predict themselves.
- **Empirical calibration over published constants**: scale factors (`WOBA_SCALE`, `RUNS_PER_WIN`, `RA9_replacement`) are fit directly from 2021–2025 data rather than assumed from external sabermetric literature.
- **Every feature earns its place**: both feature sets went through a p-value + VIF review; features that weren't statistically significant or were fully redundant with another feature were cut (`ev`, `maxev`, `pull_pct`, `sb_pct` on the hitting side), and features that passed the review were kept even when VIF was elevated for expected structural reasons (the pitcher batted-ball features).
- **Linear regression chosen on merit, not convention**: both pipelines tested linear regression against Random Forest and XGBoost, and linear regression won on held-out R² in both cases.
- **Held-out validation**: 2021–2025 data is used for training and in-sample validation of the calibration; new/future data is scored as a pure test set via the separate `xWAR_test_new_data.ipynb` notebook, never touching `.fit()`.

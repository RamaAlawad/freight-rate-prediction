# Freight Rate Prediction

Predicts `posted_rate` for 12,000 loads (Nov-Dec 2025) from 48,000 labelled loads (Jan-Oct 2025).

## Run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# put the four assessment CSVs in data/ with these names:
#   train_test.csv, validation.csv, validation_predictions_template.csv, december_chart_inputs.csv
python run_all.py        # trains, then validates with score.py (about 1-2 minutes)
```

Or step by step:

```bash
python train.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python eda.py            # optional: data study + EDA plots saved to outputs/figures/eda/
```

`train.py` writes:
- `validation_predictions.csv` (`load_id,predicted_rate`)
- `data/december_chart_inputs.csv` with `predicted_rate` filled in
- `outputs/metrics.json` and `outputs/figures/` (backtest numbers and plots used in the report)

`score.py` (provided by Spotter) then validates both files and creates `scorer_results/candidate_december.png`.

## Approach in short

- **Time-based validation.** The labelled data is Jan-Oct and the validation set is Nov-Dec, so a random split would leak the future. I use a rolling-origin backtest: train on the past, test on the following two months (three folds).
- **Cleaning.** Negative weights are sign errors (`abs`), missing weights get the median plus a missing flag, missing `market_index` is filled with that day's median. About 1.5% of labels are corrupted (rate off by roughly 0.2-0.5x or 2.3-5x); the model uses an L1 objective so they barely influence it.
- **No city identity features.** 8 cities in the validation set never appear in training (about 6% of rows), so the model uses distance, coordinates, equipment, weight and market features instead.
- **Model.** LightGBM on `log(posted_rate)` with an L1 objective, plus a small level correction (median recent residual, +2.1%) for the upward drift seen in Aug-Oct.
- **December chart.** `december_chart_inputs.csv` has no market columns, so the per-date `market_index` / `quote_signal` values from `validation.csv` (inputs, not labels) are used for those dates.

See `report.docx` for the details and backtest results.

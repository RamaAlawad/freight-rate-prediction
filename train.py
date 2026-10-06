"""Freight rate prediction pipeline.

Run:  python train.py
Outputs (in repo root / outputs/):
    validation_predictions.csv          final predictions for the 12,000 validation loads
    data/december_chart_inputs.csv      filled in place with predicted_rate
    outputs/metrics.json                backtest results used in the report
    outputs/figures/*.png               EDA / diagnostic figures used in the report
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
SEED = 42

# Residual (log space) beyond which a training label is treated as corrupted.
LABEL_OUTLIER_LOG = 0.30

FEATURES = [
    "log_dist", "dist", "hav", "dist_ratio", "equip",
    "weight", "weight_missing",
    "mi", "mi_day", "qs", "qs_day",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
]

PARAMS = dict(
    objective="regression_l1",   # robust to the corrupted-label outliers
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
    random_state=SEED,
)


# ---------------------------------------------------------------------------
# Cleaning + features
# --------------------------------------------------------------------------- 
def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame, weight_median: float, day_mi: pd.Series | None = None,
                   day_qs: pd.Series | None = None) -> pd.DataFrame:
    
    """Build features for model training or prediction."""

    X = pd.DataFrame(index=df.index)
    X["log_dist"] = np.log(df["distance"])
    X["dist"] = df["distance"]
    for c in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        X[c] = df[c]
    hav = haversine_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    X["hav"] = hav
    X["dist_ratio"] = df["distance"] / np.maximum(hav, 1.0)
    X["equip"] = df["equipment"].map({"Dry Van": 0, "Flatbed": 1, "Reefer": 2})

    # Data quality: negative weights are sign errors -> abs(); NaN -> median + flag
    w = df["weight"].abs()
    X["weight_missing"] = w.isna().astype(int)
    X["weight"] = w.fillna(weight_median)

    if day_mi is not None:
        X["mi_day"] = df["date"].map(day_mi)
        X["mi"] = X["mi_day"]
        X["qs_day"] = df["date"].map(day_qs)
        X["qs"] = X["qs_day"]
    else:
        mi = df["market_index"]
        X["mi_day"] = mi.groupby(df["date"]).transform("median")   # market_index is ~daily level
        X["mi"] = mi.fillna(X["mi_day"])                            # NaN -> that day's median
        X["qs"] = df["quote_signal"]
        X["qs_day"] = df["quote_signal"].groupby(df["date"]).transform("mean")
    return X


def fit(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**PARAMS).fit(X[FEATURES], y)


def metrics(y_true, y_pred) -> dict:
    err = np.abs(y_true - y_pred)
    return {
        "MAE": float(err.mean()),
        "MAPE_%": float((err / y_true).mean() * 100),
        "MedAPE_%": float(np.median(err / y_true) * 100),
        "RMSE": float(np.sqrt(((y_true - y_pred) ** 2).mean())),
    }


def month_block_oof(X, y, month, mask) -> pd.Series:
    """Out-of-fold log predictions, one calendar month held out at a time ."""
    oof = pd.Series(np.nan, index=X.index)
    for m in month[mask].unique():
        held = mask & (month == m)
        model = fit(X[mask & ~held], y[mask & ~held])
        oof[held] = model.predict(X.loc[held, FEATURES])
    return oof


def recent_offset(resid: pd.Series, day: pd.Series, window: int = 56) -> float:
    """Median log-residual over the last `window` days of the training period (outliers removed)."""
    keep = resid.abs() < LABEL_OUTLIER_LOG
    recent = day >= day.max() - (window - 1)
    return float(resid[keep & recent].median())


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def backtest(train: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> dict:

    """Rolling-origin backtest: always train on the past, test on the next two months."""
    
    month = train["date"].dt.to_period("M")
    day = (train["date"] - train["date"].min()).dt.days
    folds = [("2025-05-01", "2025-07-01"), ("2025-07-01", "2025-09-01"), ("2025-09-01", "2025-11-01")]
    results = {}
    for start, end in folds:
        tr = train["date"] < start
        te = (train["date"] >= start) & (train["date"] < end)
        y_te = train.loc[te, "posted_rate"].values

        # baseline: median rate-per-mile by equipment
        rpm = (train.posted_rate / train.distance)[tr].groupby(train.equipment[tr]).median()
        base_pred = (train.distance[te] * train.equipment[te].map(rpm)).values

        model = fit(X[tr], y[tr])
        pred = np.exp(model.predict(X.loc[te, FEATURES]))

        oof = month_block_oof(X, y, month, tr)
        off = recent_offset((y - oof)[tr], day[tr])
        pred_off = pred * np.exp(off)

        results[f"{start}..{end}"] = {
            "baseline_rate_per_mile": metrics(y_te, base_pred),
            "lightgbm": metrics(y_te, pred),
            "lightgbm_plus_recent_offset": metrics(y_te, pred_off),
            "offset_log": off,
        }
        print(f"fold {start}..{end}: baseline MAPE {results[f'{start}..{end}']['baseline_rate_per_mile']['MAPE_%']:.2f}% | "
              f"LGBM {results[f'{start}..{end}']['lightgbm']['MAPE_%']:.2f}% | "
              f"+offset {results[f'{start}..{end}']['lightgbm_plus_recent_offset']['MAPE_%']:.2f}%")
    return results


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(train: pd.DataFrame, backtest_res: dict, dec: pd.DataFrame, val: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    teal = "#064A56"

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    rpm = train.posted_rate / train.distance
    ax[0].hist(rpm.clip(upper=6), bins=120, color=teal)
    ax[0].set_title("Rate per mile (clipped at $6)")
    ax[0].set_xlabel("$/mile")
    sample = train.sample(6000, random_state=0)
    ax[1].scatter(sample.distance, sample.posted_rate, s=3, alpha=0.4, color=teal)
    ax[1].set_title("posted_rate vs distance")
    ax[1].set_xlabel("miles")
    ax[1].set_ylabel("$")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "eda_target.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
    d = pd.concat([train[["date", "market_index"]], val[["date", "market_index"]]])
    daily = d.groupby("date").market_index.median()
    ax.plot(daily.index, daily.values, color=teal, lw=1.4)
    ax.axvline(pd.Timestamp("2025-11-01"), color="#C0392B", ls="--", lw=1)
    ax.text(pd.Timestamp("2025-11-03"), daily.max() * 0.97, "validation period\n(no labels)", color="#C0392B", fontsize=8)
    ax.set_title("Daily market_index: train (Jan-Oct) and validation (Nov-Dec)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "market_index_timeline.png")
    plt.close(fig)

    folds = list(backtest_res)
    names = [("baseline_rate_per_mile", "Baseline $/mile"), ("lightgbm", "LightGBM"),
             ("lightgbm_plus_recent_offset", "LightGBM + recent offset")]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    width = 0.26
    for i, (key, label) in enumerate(names):
        vals = [backtest_res[f][key]["MAPE_%"] for f in folds]
        ax.bar(np.arange(len(folds)) + (i - 1) * width, vals, width, label=label,
               color=["#9DAFB3", teal, "#2A9D8F"][i])
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f.replace("..", " to ") for f in folds], fontsize=8)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Rolling-origin backtest (train on past, test on next 2 months)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "backtest_mape.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    val = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    dec = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])

    print(f"train {train.shape}, {train.date.min().date()}..{train.date.max().date()} | "
          f"validation {val.shape}, {val.date.min().date()}..{val.date.max().date()}")

    weight_median = train["weight"].abs().median()
    X = build_features(train, weight_median)
    y = np.log(train["posted_rate"])
    Xv = build_features(val, weight_median)

    # 1) honest, time-aware validation
    bt = backtest(train, X, y)

    # 2) final model on all labelled data
    month = train["date"].dt.to_period("M")
    day = (train["date"] - train["date"].min()).dt.days
    oof = month_block_oof(X, y, month, pd.Series(True, index=train.index))
    offset = recent_offset(y - oof, day)
    print(f"recent-residual offset (log) = {offset:+.4f}")
    model = fit(X, y)

    pred_val = np.exp(model.predict(Xv[FEATURES]) + offset)

    # 3) validation predictions in template order
    out = template[["load_id"]].merge(pd.DataFrame({"load_id": val.load_id, "predicted_rate": pred_val.round(2)}),
                                      on="load_id", how="left")
    assert out.predicted_rate.notna().all() and len(out) == 12_000 and (out.predicted_rate > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)

    # 4) December chart: market columns are absent from that file, so use the per-date
    #    market values given in validation.csv (known inputs, not labels).
    day_mi = val.groupby("date").market_index.median()
    day_qs = val.groupby("date").quote_signal.mean()
    lat = train.drop_duplicates("pickup").set_index("pickup")[["pickup_lat", "pickup_lon"]]
    lat_d = train.drop_duplicates("delivery").set_index("delivery")[["delivery_lat", "delivery_lon"]]
    d = dec.copy()
    d["pickup_lat"] = d.pickup.map(lat.pickup_lat)
    d["pickup_lon"] = d.pickup.map(lat.pickup_lon)
    d["delivery_lat"] = d.delivery.map(lat_d.delivery_lat)
    d["delivery_lon"] = d.delivery.map(lat_d.delivery_lon)
    Xd = build_features(d, weight_median, day_mi=day_mi, day_qs=day_qs)
    dec_pred = np.exp(model.predict(Xd[FEATURES]) + offset).round(2)
    dec_out = pd.read_csv(DATA / "december_chart_inputs.csv")   # keep original dtypes/format
    dec_out["predicted_rate"] = dec_pred
    dec_out.to_csv(DATA / "december_chart_inputs.csv", index=False)
    print(f"December predicted range: ${dec_pred.min():.0f} - ${dec_pred.max():.0f}")

    # 5) diagnostics
    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps({
        "backtest": bt,
        "recent_offset_log": offset,
        "feature_importance_gain_pct": (imp / imp.sum() * 100).round(2).to_dict(),
        "data_quality": {
            "negative_weight_rows": int((train.weight < 0).sum()),
            "missing_weight_rows": int(train.weight.isna().sum()),
            "missing_market_index_rows": int(train.market_index.isna().sum()),
            "validation_rows_with_unseen_pickup_city": int((~val.pickup.isin(train.pickup)).sum()),
            "validation_rows_with_unseen_delivery_city": int((~val.delivery.isin(train.delivery)).sum()),
            "extreme_label_rows_oof_gt_0.3_log": int(((y - oof).abs() > LABEL_OUTLIER_LOG).sum()),
        },
        "validation_pred_summary": pd.Series(pred_val).describe().round(2).to_dict(),
    }, indent=2))
    make_figures(train, bt, dec_out, val)
    print("done")


if __name__ == "__main__":
    main()

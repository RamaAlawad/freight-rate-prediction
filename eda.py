"""Exploratory data analysis (optional, independent of the model).

Run:  python eda.py
Prints a data study of train and validation and saves every plot to outputs/figures/eda/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

ROOT = Path(__file__).parent
DATA = ROOT / "data"
EDA_DIR = ROOT / "outputs" / "figures" / "eda"

_counter = 0


def save_fig(name: str) -> None:
    """Save the current figure as NN_name.png and close it (avoids memory build-up)."""
    global _counter
    _counter += 1
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(EDA_DIR / f"{_counter:02d}_{name}.png", dpi=110, bbox_inches="tight")
    plt.close("all")


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------- #
# 1) Data study (text report)
# --------------------------------------------------------------------------- #
def study_data(df: pd.DataFrame, name: str, target_col: str | None = None) -> None:
    header(f"DATA STUDY: {name}")
    print(f"Shape: {df.shape}")

    print("\n--- Column types ---")
    print(df.dtypes)

    missing = df.isna().sum()
    report = pd.DataFrame({"missing_count": missing, "missing_pct": (missing / len(df) * 100).round(2)})
    print("\n--- Missing values ---")
    print(report[report.missing_count > 0].sort_values("missing_count", ascending=False))

    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    print("\n--- Numeric summary ---")
    print(df.select_dtypes(include=np.number).describe().T)

    print("\n--- Categorical columns (top 5 values; load_id skipped, it is unique per row) ---")
    for col in df.select_dtypes(exclude=["number", "datetime"]).columns:
        if col == "load_id":
            continue
        print(f"\n{col} ({df[col].nunique()} unique):")
        print(df[col].value_counts().head(5))

    if target_col and target_col in df.columns:
        t = df[target_col]
        print(f"\n--- Target: {target_col} ---")
        print(f"min {t.min():,.2f} | median {t.median():,.2f} | mean {t.mean():,.2f} | max {t.max():,.2f}")
        print("Quantiles:")
        print(t.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]))

    if "weight" in df.columns:
        print(f"\nNegative weights: {(df['weight'] < 0).sum():,}")
    if "distance" in df.columns:
        print(f"Non-positive distances: {(df['distance'] <= 0).sum():,}")
    if "date" in df.columns:
        print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")


# --------------------------------------------------------------------------- #
# 2) Plots
# --------------------------------------------------------------------------- #
def perform_eda(df: pd.DataFrame) -> None:
    numerical = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.select_dtypes(exclude=["number", "datetime"]).columns if c != "load_id"]

    header("NUMERICAL DISTRIBUTIONS")
    for col in numerical:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=col, kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        save_fig(f"hist_{col}")

    header("OUTLIERS (boxplots)")
    for col in numerical:
        plt.figure(figsize=(8, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        save_fig(f"box_{col}")

    header("CATEGORICAL DISTRIBUTIONS")
    for col in categorical:
        plt.figure(figsize=(12, 5))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=90 if df[col].nunique() > 10 else 0)
        save_fig(f"count_{col}")

    header("CORRELATION")
    corr = df[numerical].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix")
    save_fig("correlation")
    print(corr.round(2))

    header("PAIRWISE RELATIONSHIPS")
    selected = numerical[:6]  # defined once, before use
    if len(numerical) > 6:
        print(f"{len(numerical)} numerical columns; showing the first 6: {selected}")
    if len(selected) >= 2:
        sns.pairplot(df[selected].sample(min(5000, len(df)), random_state=0), corner=True, diag_kind="hist")
        save_fig("pairplot")

    header("EQUIPMENT VS NUMERICAL FEATURES")
    if "equipment" in df.columns:
        for col in ("distance", "weight", "market_index"):
            if col in df.columns:
                plt.figure(figsize=(8, 5))
                sns.boxplot(data=df, x="equipment", y=col)
                plt.title(f"{col} by equipment")
                save_fig(f"equipment_{col}")

    header("TIME-BASED ANALYSIS")
    if "date" in df.columns:
        ts = df.set_index("date")
        plt.figure(figsize=(12, 5))
        ts.resample("W").size().plot()
        plt.title("Loads per week")
        save_fig("loads_per_week")

        for col in ("market_index", "posted_rate"):
            if col in ts.columns:
                plt.figure(figsize=(12, 5))
                ts[col].resample("W").mean().plot()
                plt.title(f"Weekly mean of {col}")
                save_fig(f"weekly_{col}")

    header("SKEWNESS")
    skew = df[numerical].skew().sort_values(ascending=False)
    for col, v in skew.items():
        label = "highly skewed" if abs(v) > 1 else "moderately skewed" if abs(v) > 0.5 else "approximately symmetric"
        print(f"{col}: {v:.2f} -> {label}")

    print(f"\nPlots saved to {EDA_DIR}")


if __name__ == "__main__":
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    val = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    study_data(train, "TRAIN", target_col="posted_rate")
    study_data(val, "VALIDATION")
    perform_eda(train)
    print("\nEDA done")

"""Run the whole pipeline with one command:  python run_all.py

1) train.py  -> validation_predictions.csv + filled data/december_chart_inputs.csv
2) score.py  -> validates both files and creates scorer_results/candidate_december.png
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run(*args: str) -> None:
    print("\n>>", " ".join(args))
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    run("train.py")
    run(
        "score.py",
        "--predictions", "validation_predictions.csv",
        "--december-predictions", "data/december_chart_inputs.csv",
    )
    print("\nAll steps finished.")

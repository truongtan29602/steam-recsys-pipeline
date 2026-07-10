"""Generate plots for Task 3 model comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_results(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for split, items in raw.items():
        for item in items:
            row = dict(item)
            row["split"] = split
            rows.append(row)
    return pd.DataFrame(rows)


def plot_bars(df: pd.DataFrame, metric: str, out: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=df, x="model", y=metric, hue="split")
    plt.title(metric.replace("_", " ").upper())
    plt.xlabel("")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_latency(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=df, x="model", y="recommendation_time_sec", hue="split")
    plt.title("Recommendation latency")
    plt.xlabel("")
    plt.ylabel("seconds")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_tradeoff(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="coverage@20",
        y="recall@20",
        hue="model",
        style="split",
        s=120,
    )
    for _, row in df.iterrows():
        plt.text(row["coverage@20"], row["recall@20"], f" {row['model']} / {row['split']}", fontsize=8)
    plt.title("Coverage vs Recall")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Task 3 evaluation results.")
    parser.add_argument("--results", type=Path, default=Path("outputs/task3/task3_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task3/plots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(args.results)

    plot_bars(df, "recall@20", args.output_dir / "recall_at_20.png")
    plot_bars(df, "ndcg@10", args.output_dir / "ndcg_at_10.png")
    plot_bars(df, "coverage@20", args.output_dir / "coverage_at_20.png")
    plot_latency(df, args.output_dir / "latency.png")
    plot_tradeoff(df, args.output_dir / "coverage_vs_recall.png")


if __name__ == "__main__":
    main()

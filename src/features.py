"""
CreatorSignal Chicago - Week 3 Deliverable
Feature engineering library: one-click build from the cleaned SQL table to a
model-ready dataset, plus a documented feature dictionary.

Run: python src/features.py
Benchmark target: one-click dataset build from raw exports
"""
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "creatorsignal.db"
OUT_PATH = BASE / "data" / "model_ready_dataset.csv"

FEATURE_DICTIONARY = {
    "length_seconds": "Post/video duration in seconds",
    "post_hour": "Hour of day posted (0-23)",
    "is_weekend": "1 if posted Sat/Sun, else 0",
    "is_prime_time": "1 if posted 5-8pm local, else 0",
    "theme_*": "One-hot encoded content theme tag",
    "platform_*": "One-hot encoded platform (TikTok/Instagram/YouTube)",
    "creator_avg_engagement_rate": "Creator's trailing historical average engagement rate (excl. current post)",
    "performance_tier": "TARGET - low / expected / breakout, derived from engagement_rate tercile within platform",
}


def build_dataset():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM creator_posts", conn, parse_dates=["post_date"])
    conn.close()

    df = df.sort_values(["creator_handle", "post_date"])

    # prime-time flag
    df["is_prime_time"] = df["post_hour"].between(17, 20).astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)

    # trailing creator-level engagement baseline (avoids leakage: excludes current post)
    df["creator_avg_engagement_rate"] = (
        df.groupby("creator_handle")["engagement_rate"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["creator_avg_engagement_rate"] = df["creator_avg_engagement_rate"].fillna(
        df["engagement_rate"].median()
    )

    # target: performance tier via platform-level terciles on engagement_rate
    def tier_within_platform(group):
        q1, q2 = group["engagement_rate"].quantile([1 / 3, 2 / 3])
        def tier(v):
            if v <= q1:
                return "low"
            elif v <= q2:
                return "expected"
            return "breakout"
        return group["engagement_rate"].apply(tier)

    df["performance_tier"] = (
        df.groupby("platform", group_keys=False).apply(tier_within_platform)
    )

    # one-hot encode categoricals
    theme_dummies = pd.get_dummies(df["theme_tag"], prefix="theme")
    platform_dummies = pd.get_dummies(df["platform"], prefix="platform")

    model_df = pd.concat([
        df[["creator_handle", "post_id", "post_date", "length_seconds", "post_hour",
            "is_weekend", "is_prime_time", "creator_avg_engagement_rate",
            "engagement_rate", "performance_tier"]],
        theme_dummies, platform_dummies,
    ], axis=1)

    model_df.to_csv(OUT_PATH, index=False)
    return model_df


def main():
    df = build_dataset()
    print(f"Built model-ready dataset: {df.shape[0]} rows x {df.shape[1]} cols -> {OUT_PATH}")
    print("\nFeature dictionary:")
    for k, v in FEATURE_DICTIONARY.items():
        print(f"  {k:32s} {v}")
    print("\nTarget class balance:")
    print(df["performance_tier"].value_counts())
    return df


if __name__ == "__main__":
    main()

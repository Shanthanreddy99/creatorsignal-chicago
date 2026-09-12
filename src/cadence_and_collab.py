"""
CreatorSignal Chicago - Week 4 & Week 6 Deliverables
1. Posting Cadence Optimizer: model-based best posting windows per platform
2. Collab Match Engine: creator-to-creator similarity scoring

Run: python src/cadence_and_collab.py
Benchmark target (Week 6): generates top 5 collaboration candidates per creator
"""
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "creatorsignal.db"
OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_posts():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM creator_posts", conn, parse_dates=["post_date"])
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Posting Cadence Optimizer
# ---------------------------------------------------------------------------
def cadence_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for platform, g in df.groupby("platform"):
        hourly = g.groupby("post_hour")["engagement_rate"].agg(["mean", "count"])
        hourly = hourly[hourly["count"] >= 3]  # only trust windows with enough samples
        top_hours = hourly.sort_values("mean", ascending=False).head(3)

        daily = g.groupby("day_of_week")["engagement_rate"].mean().sort_values(ascending=False)

        # cadence impact: does posting frequency correlate with per-post engagement?
        weekly_counts = g.set_index("post_date").resample("W")["post_id"].count()
        weekly_eng = g.set_index("post_date").resample("W")["engagement_rate"].mean()
        cadence_df = pd.concat([weekly_counts, weekly_eng], axis=1).dropna()
        cadence_df.columns = ["posts_per_week", "avg_engagement"]
        if len(cadence_df) >= 5 and cadence_df["posts_per_week"].std() > 0:
            cadence_corr = cadence_df["posts_per_week"].corr(cadence_df["avg_engagement"])
        else:
            cadence_corr = None

        for hour, r in top_hours.iterrows():
            rows.append({
                "platform": platform,
                "recommended_hour": int(hour),
                "avg_engagement_rate": round(r["mean"], 4),
                "sample_size": int(r["count"]),
                "best_day": daily.index[0],
                "cadence_vs_engagement_corr": None if cadence_corr is None else round(cadence_corr, 3),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Collab Match Engine
# ---------------------------------------------------------------------------
def build_creator_profiles(df: pd.DataFrame) -> pd.DataFrame:
    theme_dist = pd.crosstab(df["creator_handle"], df["theme_tag"], normalize="index")

    engagement_profile = df.groupby("creator_handle").agg(
        avg_engagement_rate=("engagement_rate", "mean"),
        avg_reach=("reach", "mean"),
        posting_freq_per_month=("post_id", lambda s: len(s) / max(1, df["post_date"].dt.to_period("M").nunique())),
        weekend_share=("is_weekend", "mean"),
    )
    # normalize engagement profile to comparable 0-1 scale for cosine similarity
    eng_norm = (engagement_profile - engagement_profile.min()) / (
        engagement_profile.max() - engagement_profile.min() + 1e-9
    )

    profiles = theme_dist.join(eng_norm.add_prefix("eng_"))
    return profiles.fillna(0)


def collab_matches(profiles: pd.DataFrame, df: pd.DataFrame, top_n=5) -> pd.DataFrame:
    sim_matrix = cosine_similarity(profiles.values)
    handles = profiles.index.tolist()
    platform_map = df.groupby("creator_handle")["platform"].first()

    results = []
    for i, creator in enumerate(handles):
        sims = [(handles[j], sim_matrix[i][j]) for j in range(len(handles)) if j != i]
        sims.sort(key=lambda x: x[1], reverse=True)
        for rank, (other, score) in enumerate(sims[:top_n], start=1):
            cross_platform = platform_map.get(creator) != platform_map.get(other)
            results.append({
                "creator": creator,
                "rank": rank,
                "match_candidate": other,
                "similarity_score": round(float(score), 4),
                "cross_platform_opportunity": cross_platform,
                "candidate_platform": platform_map.get(other),
            })
    return pd.DataFrame(results)


def main():
    df = load_posts()

    cadence = cadence_recommendations(df)
    cadence_path = OUT_DIR / "cadence_recommendations.csv"
    cadence.to_csv(cadence_path, index=False)
    print("Posting Cadence Recommendations:")
    print(cadence.to_string(index=False))

    profiles = build_creator_profiles(df)
    matches = collab_matches(profiles, df, top_n=5)
    matches_path = OUT_DIR / "collab_matches.csv"
    matches.to_csv(matches_path, index=False)
    print("\nTop Collab Matches (sample):")
    print(matches[matches["rank"] <= 2].to_string(index=False))

    print(f"\nWrote -> {cadence_path}")
    print(f"Wrote -> {matches_path}")
    return cadence, matches


if __name__ == "__main__":
    main()

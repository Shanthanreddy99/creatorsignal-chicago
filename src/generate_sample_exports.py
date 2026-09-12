"""
CreatorSignal Chicago - Week 1 Deliverable
Generates realistic sample creator analytics exports for pipeline development.

In production these come from creator-submitted CSV exports (per the intake
form). For this evidence-of-work build, we simulate 3 Chicago creators across
TikTok, Instagram, and YouTube with realistic noise, seasonality, and messy
raw-export quirks (duplicate rows, inconsistent casing, missing values) so the
cleaning pipeline in step 2 has real work to do.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_exports"
RAW_DIR.mkdir(parents=True, exist_ok=True)

THEMES = ["food_review", "neighborhood_guide", "day_in_life", "event_recap",
          "tutorial", "opinion_take", "collab_feature", "behind_scenes"]

CREATORS = [
    {"handle": "chi_eats_ari",     "platform": "TikTok",    "n_posts": 140, "base_reach": 8000,  "skill": 1.15},
    {"handle": "westloop_wanders", "platform": "Instagram", "n_posts": 110, "base_reach": 4200,  "skill": 0.95},
    {"handle": "loopline_marcus",  "platform": "YouTube",   "n_posts": 65,  "base_reach": 15000, "skill": 1.05},
]


def make_creator_export(handle, platform, n_posts, base_reach, skill):
    start = datetime(2025, 1, 1)
    rows = []
    for i in range(n_posts):
        post_date = start + timedelta(days=int(RNG.integers(0, 300)))
        theme = RNG.choice(THEMES, p=[0.22, 0.15, 0.13, 0.10, 0.15, 0.10, 0.08, 0.07])
        length_sec = int(RNG.normal(45 if platform == "TikTok" else 90, 20))
        length_sec = max(8, length_sec)
        hour = int(RNG.integers(6, 24))
        dow = post_date.weekday()

        # underlying signal: prime hours + weekends + certain themes perform better
        prime_hour = 1.6 if hour in (17, 18, 19, 20) else (1.15 if hour in (11, 12, 13) else 0.85)
        weekend_bonus = 1.35 if dow >= 5 else 1.0
        theme_bonus = {"food_review": 1.45, "neighborhood_guide": 1.25, "collab_feature": 1.5,
                       "day_in_life": 0.75, "event_recap": 1.1, "tutorial": 1.15,
                       "opinion_take": 0.65, "behind_scenes": 0.8}[theme]
        length_bonus = 1.2 if 25 <= length_sec <= 60 else 0.9

        noise = RNG.lognormal(mean=0, sigma=0.28)
        reach = base_reach * skill * prime_hour * weekend_bonus * theme_bonus * length_bonus * noise
        reach = max(150, reach)

        # engagement rate itself carries the same signal (not just reach),
        # since that's what the model is actually predicting downstream
        eng_signal = prime_hour * weekend_bonus * theme_bonus * length_bonus
        eng_noise = RNG.lognormal(mean=0, sigma=0.22)
        eng_multiplier = (eng_signal ** 0.6) * eng_noise

        likes = reach * min(0.14, 0.035 * eng_multiplier)
        comments = reach * min(0.02, 0.004 * eng_multiplier)
        shares = reach * min(0.03, 0.005 * eng_multiplier)
        saves = reach * min(0.025, 0.004 * eng_multiplier)

        row = {
            "creator_handle": handle if RNG.random() > 0.05 else handle.upper(),  # casing noise
            "platform": platform,
            "post_id": f"{handle[:4]}_{i:04d}",
            "post_date": post_date.strftime("%Y-%m-%d"),
            "post_hour": hour,
            "theme_tag": theme if RNG.random() > 0.08 else theme.replace("_", " ").title(),  # format noise
            "length_seconds": length_sec if RNG.random() > 0.03 else None,  # missing value noise
            "reach": int(reach),
            "likes": int(likes),
            "comments": int(comments),
            "shares": int(shares),
            "saves": int(saves),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # inject a handful of exact duplicate rows, as raw exports often have
    dupes = df.sample(n=max(1, n_posts // 40), random_state=1)
    df = pd.concat([df, dupes], ignore_index=True)
    return df


def main():
    manifest = []
    for c in CREATORS:
        df = make_creator_export(c["handle"], c["platform"], c["n_posts"], c["base_reach"], c["skill"])
        out_path = RAW_DIR / f"{c['handle']}_{c['platform'].lower()}_export.csv"
        df.to_csv(out_path, index=False)
        manifest.append({"creator": c["handle"], "platform": c["platform"],
                          "rows": len(df), "file": str(out_path)})
        print(f"  wrote {out_path.name}: {len(df)} rows")
    return manifest


if __name__ == "__main__":
    main()

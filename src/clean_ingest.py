"""
CreatorSignal Chicago - Week 2 Deliverable
Ingestion + cleaning pipeline: raw creator CSV exports -> standardized,
validated, deduplicated table stored in SQLite.

Run: python src/clean_ingest.py
Benchmark target: upload -> cleaned tables stored -> reproducible run
"""
import sqlite3
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / "data" / "raw_exports"
DB_PATH = BASE / "data" / "creatorsignal.db"

EXPECTED_COLUMNS = [
    "creator_handle", "platform", "post_id", "post_date", "post_hour",
    "theme_tag", "length_seconds", "reach", "likes", "comments", "shares", "saves",
]


def load_raw_exports():
    frames = []
    for f in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No raw exports found in {RAW_DIR}. Run generate_sample_exports.py first.")
    return pd.concat(frames, ignore_index=True)


def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"rows_in": len(df)}

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # standardize text fields
    df["creator_handle"] = df["creator_handle"].str.lower().str.strip()
    df["theme_tag"] = df["theme_tag"].str.lower().str.replace(" ", "_").str.strip()
    df["platform"] = df["platform"].str.strip()

    # types
    df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
    numeric_cols = ["post_hour", "length_seconds", "reach", "likes", "comments", "shares", "saves"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    report["rows_bad_dates"] = int(df["post_date"].isna().sum())
    df = df.dropna(subset=["post_date", "reach"])

    # impute missing length with theme-level median (documented, defensible choice)
    report["rows_missing_length"] = int(df["length_seconds"].isna().sum())
    df["length_seconds"] = df.groupby("theme_tag")["length_seconds"].transform(
        lambda s: s.fillna(s.median())
    )
    df["length_seconds"] = df["length_seconds"].fillna(df["length_seconds"].median())

    # dedupe
    before = len(df)
    df = df.drop_duplicates(subset=["creator_handle", "platform", "post_id"])
    report["duplicates_removed"] = before - len(df)

    # derived fields used downstream
    df["day_of_week"] = df["post_date"].dt.day_name()
    df["is_weekend"] = df["post_date"].dt.dayofweek >= 5
    df["engagement_rate"] = (
        (df["likes"] + df["comments"] + df["shares"] + df["saves"]) / df["reach"]
    ).round(4)

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def store(df: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("creator_posts", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_creator ON creator_posts(creator_handle)")
    conn.commit()
    conn.close()


def main():
    raw = load_raw_exports()
    clean, report = validate_and_clean(raw)
    store(clean)
    print("Cleaning report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"Stored {len(clean)} clean rows -> {DB_PATH}")
    return clean, report


if __name__ == "__main__":
    main()

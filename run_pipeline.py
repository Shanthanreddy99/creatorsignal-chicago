"""
CreatorSignal Chicago - Full Pipeline Runner
Runs the entire MVP end-to-end: raw exports -> clean SQL -> features ->
model training -> cadence + collab engines -> CreatorSignal Report PDF.

Run: python run_pipeline.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import generate_sample_exports
import clean_ingest
import features
import model
import cadence_and_collab
import report


def step(name, fn):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    t0 = time.time()
    result = fn()
    print(f"[{time.time()-t0:.2f}s]")
    return result


def main():
    step("WEEK 1 — Raw creator export ingestion (sample data)", generate_sample_exports.main)
    step("WEEK 2 — SQL cleaning & standardization pipeline", clean_ingest.main)
    step("WEEK 3 — Feature engineering", features.main)
    step("WEEK 4-5 — Baseline + tuned classifier, evaluation", model.run)
    step("WEEK 4 & 6 — Cadence optimizer + collab match engine", cadence_and_collab.main)
    step("WEEK 7 — Dashboard + CreatorSignal Report", report.main)
    print("\nPipeline complete. See /outputs for all artifacts.")


if __name__ == "__main__":
    main()

# CreatorSignal Chicago

Applied ML decision-support pipeline for Chicago content creators — content performance forecasting, posting cadence optimization, and audience-overlap collaboration matching.

**This is not a content generator. This is decision support.**

## Live Demo

🔗 **[creatorsignal-chicago.streamlit.app](https://creatorsignal-chicago.streamlit.app)** *(update this link after deploying)*

## What it does

CreatorSignal takes a creator's own analytics exports and turns them into three actionable answers:
- **What** to post about (content theme performance analysis)
- **When** to post it (cadence optimization with minimum sample thresholds)
- **Who** to collaborate with (cosine similarity matching across content and engagement profiles)

## Pipeline stages

| Stage | What it does | Module |
|---|---|---|
| 1. Data generation | Sample creator exports with realistic noise | `src/generate_sample_exports.py` |
| 2. Cleaning + SQL | Standardize, deduplicate, store in SQLite | `src/clean_ingest.py` |
| 3. Feature engineering | Model-ready signals with documented dictionary | `src/features.py` |
| 4–5. Modeling | Baseline (logistic regression) + tuned (random forest) with honest comparison | `src/model.py` |
| 6. Cadence + Collab | Posting time optimizer + collaboration match engine | `src/cadence_and_collab.py` |
| 7. Dashboard | Interactive Streamlit web app | `app.py` |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The pipeline runs automatically on first load (~15 seconds), generating sample data for three Chicago creators across TikTok, Instagram, and YouTube.

## Tech stack

Python · Pandas · scikit-learn · Plotly · Streamlit · SQLite

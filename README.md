# Universal Bank Personal Loan Dashboard

This Streamlit dashboard is built for the Universal Bank assignment using `UniversalBank.csv`.

## What it does
- Predicts `Personal Loan` acceptance with Decision Tree, Random Forest, and Gradient Boosting.
- Excludes `ZIP Code` and `ID` from modeling.
- Covers descriptive, diagnostic, predictive, and prescriptive analytics.
- Recommends cross-sell offers for likely loan acceptors.
- Creates simple customer personas for campaign targeting.

## Files
- `app.py` — main Streamlit app
- `requirements.txt` — dependencies
- `.streamlit/config.toml` — theme config
- `UniversalBank.csv` — dataset

## Project structure
```bash
repo/
├── app.py
├── UniversalBank.csv
├── requirements.txt
└── .streamlit/
    └── config.toml
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Upload `app.py`, `UniversalBank.csv`, and `requirements.txt` to your GitHub repo.
2. Create a `.streamlit` folder and place `config.toml` inside it.
3. In Streamlit Cloud, set the main file path to `app.py`.

## Data notes
- Rows: 5,000
- Columns: 14
- Target: `Personal Loan`
- Ignored in model: `ID`, `ZIP Code`
- Negative `Experience` values are clipped to 0 in the app.

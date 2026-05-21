# RST Portfolio - Streamlit App

The web front end for the portfolio forecasting tool. Reads financial actuals
from S3 and renders the internal RST proforma view.

## Setup (one time)

1. Create a Python virtual environment (recommended):

   ```bash
   cd app
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up credentials:

   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

   Then open `.streamlit/secrets.toml` and paste your AWS Access Key ID
   and Secret Access Key. The IAM user should have read access to
   `s3://testbucketcomproject/historic_actuals.csv`.

   The real `secrets.toml` is gitignored - it never gets committed.

## Run the app

```bash
cd app
streamlit run app.py
```

Streamlit will open at `http://localhost:8501`.

## Current status (v0.1)

Skeleton only. Confirms Streamlit runs. No S3 connection, no proforma yet.

Next iteration will add:
- S3 loader (`load_actuals()`)
- Property + period selectors in the sidebar
- Proforma renderer matching the internal RST format

## Folder layout

```
app/
├── app.py                            Streamlit entry point
├── requirements.txt                  Python deps
├── README.md                         This file
└── .streamlit/
    ├── secrets.toml.example          Template - safe to commit
    └── secrets.toml                  Real creds - GITIGNORED, never commit
```

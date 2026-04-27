# Projet_d_etude_G7
Before running, make sure you have 1. filled with your own secret variables and 2. renamed correctly the following files:
- `parameters_nlp_cleaning_template.yml`
- `.env_template` in the root of the project
Also review `config/app_config.json` for non-secret settings (limits, handles, alert threshold).

Always be in your virtual environment; run: `.venv\Scripts\activate`
Before running the NLP pipeline the first time, install TextBlob/NLTK corpora:
`python -m textblob.download_corpora`
`python -c "import nltk; nltk.download('punkt_tab')"`
To run the containers (airflow and postres):
`cd airflow`
`docker compose up -d`
Access Airflow at: http://localhost:8080/

To run kedro pipeline:
`cd bluesky-pipeline`
`kedro run --pipeline nlp_cleaning`

To run collectors:
`python src/blueskyToMongoBackfill.py`

To run Kedro with energy tracking:
`python tools/run_kedro_with_energy.py`

## API (FastAPI)
`pip install -r requirements_app.txt`
`uvicorn apps.api.main:app --reload --port 8000`

## Dashboard (Streamlit)
`pip install -r requirements_app.txt`
`streamlit run apps/dashboard/app.py`

## Monitoring (Grafana + Prometheus)
`cd monitoring`
`docker compose up -d`
Open Grafana at http://localhost:3000 (admin / admin).


## Documentation

### Kedro : 
https://docs.kedro.org/en/1.0.0/getting-started/course/

### Airflow :
https://airflow.apache.org/docs/

### Dash : 
https://dash.plotly.com/

### Streamlit :
https://docs.streamlit.io/

# Projet_d_etude_G7
Before running, make sure you have 1. filled with your own secret variables and 2. renamed correctly the following files:
- `parameters_nlp_cleaning_template.yml`
- `.env_template` in the root of the project

Always be in your virtual environment; run: `.venv\Scripts\activate`
To run the containers (airflow and postres):
`cd airflow`
`docker compose up -d`
Access Airflow at: http://localhost:8080/

To run kedro pipeline:
`cd bluesky-pipeline`
`kedro run --pipeline nlp_cleaning`


## Documentation

### Kedro : 
https://docs.kedro.org/en/1.0.0/getting-started/course/

### Airflow :
https://airflow.apache.org/docs/

### Dash : 
https://dash.plotly.com/

### Streamlit :
https://docs.streamlit.io/

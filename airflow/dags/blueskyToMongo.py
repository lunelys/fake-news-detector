from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from insertToMongo import main

default_args = {
    'owner': 'lunelys',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='bluesky_to_mongo',
    default_args=default_args,
    description='Récupère les posts Bluesky et les insère dans MongoDB',
    schedule='0 * * * *',  # Toutes les heures
    start_date=datetime(2026, 1, 1), # Date dans le passé, sinon le DAG ne se lance pas
    catchup=False,
) as dag:

    run_pipeline = PythonOperator(
        task_id='run_bluesky_pipeline',
        python_callable=main
    )

    run_pipeline
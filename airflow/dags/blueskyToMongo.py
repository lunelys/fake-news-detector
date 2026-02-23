from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from blueskyToMongoBackfill import main

default_args = {
    'owner': 'lunelys',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='bluesky_to_mongo',
    default_args=default_args,
    description='Fetch Bluesky posts and insert them into MongoDB',
    schedule='0 * * * *',  # Every hour
    start_date=datetime(2026, 1, 1),  # in the past to trigger immediately on deployment
    catchup=False,
) as dag:

    run_pipeline = PythonOperator(
        task_id='run_bluesky_pipeline',
        python_callable=main
    )

    run_pipeline
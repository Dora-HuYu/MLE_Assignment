import sys
import os
sys.path.append("/opt/airflow/src")

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
from utils.check_feature_ready import check_source_data_bronze_feature
from utils.check_label_ready import check_source_data_bronze_label

default_args = {
    'owner': 'HuYu',
    'depends_on_past': False,
    'retries': 0
}

with DAG(
    dag_id="MLE-DAG",
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    catchup=True,
) as dag:

    # ---- Data Checking ----
    dep_check_source_data_bronze_label = PythonOperator(
        task_id="dep_check_source_data_bronze_label",
        python_callable=check_source_data_bronze_label
    )

    dep_check_source_feature_data = PythonOperator(
        task_id="dep_check_source_feature_data",
        python_callable=check_source_data_bronze_feature
    )

    # ---- Feature Store ----
    feature_bronze_table = BashOperator(
        task_id="feature_bronze_table",
        bash_command="docker exec inference1 env python /opt/airflow/src/data_pipeline/data_processing_bronze_table.py"
    )

    feature_silver_table = BashOperator(
        task_id="feature_silver_table",
        bash_command="docker exec inference1 env python /opt/airflow/src/data_pipeline/data_processing_silver_table.py"
    )

    feature_gold_table = BashOperator(
        task_id="feature_gold_table",
        bash_command="docker exec inference1 env python /opt/airflow/src/data_pipeline/data_processing_gold_table.py"
    )

    feature_store_completed = BashOperator(
        task_id="feature_store_completed",
        bash_command="echo 'Feature store completed'"
    )

    dep_check_source_feature_data  >> feature_bronze_table >> feature_silver_table >> feature_gold_table >> feature_store_completed

    # ---- Label Store ----
    bronze_label_store = BashOperator(
        task_id="bronze_label_store",
        bash_command="docker exec inference1 env python /opt/airflow/src/data_pipeline/data_processing_bronze_table.py"
    )

    silver_label_store = BashOperator(
        task_id="silver_label_store",
        bash_command="docker exec inference1 env python /opt/airflow/src/data_pipeline/data_processing_silver_table.py"
    )

    gold_label_store = BashOperator(
        task_id="gold_label_store",
        bash_command="docker exec inference1 env python /opt/airflow/src/data_pipeline/data_processing_gold_table.py"
    )

    label_store_completed = BashOperator(
        task_id="label_store_completed",
        bash_command="echo 'Label store completed'"
    )

    dep_check_source_data_bronze_label >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed

    # ---- Model Train ----
    model_train_start = BashOperator(
        task_id="model_train_start",
        bash_command="echo 'Model training started'"
    )

    model_train = BashOperator(
        task_id="model_train",
        bash_command="" 
            "echo 'Running training for snapshot: 2024-09-01' && "
            "docker exec inference1 env python /opt/airflow/src/model_train/model_training.py --snapshotdate 2024-09-01"
    )

    model_train_completed = BashOperator(
        task_id="model_train_completed",
        bash_command="echo 'Model training completed'"
    )

    [feature_store_completed, label_store_completed] >> model_train_start
    model_train_start >> model_train >> model_train_completed

    # ---- Inference ----
    model_inference_start = BashOperator(
        task_id="model_inference_start",
        bash_command="echo 'Model inference started'"
    )

    model_inference = BashOperator(
        task_id="model_inference",
        bash_command="""
        docker exec inference1 \
        bash -c 'PYTHONPATH=/opt/airflow/src \
        python /opt/airflow/src/model_inference/model_inference.py \
        --snapshotdate 2024-10-01 \
        --modelname credit_model_2024_09_01.pkl'
        """
    )


    model_inference_completed = BashOperator(
        task_id="model_inference_completed",
        bash_command="echo 'Model inference completed'"
    )

    feature_store_completed >> model_inference_start >> model_inference >> model_inference_completed

    # ---- Monitor ----
    model_monitor_start = BashOperator(
        task_id="model_monitor_start",
        bash_command="echo 'Model monitoring started'"
    )

    model_monitor = BashOperator(
        task_id="model_monitor",
        bash_command="""
            docker exec inference1 env PYTHONPATH=/opt/airflow \
            python /opt/airflow/src/model_monitor/model_monitor.py \
            --snapshotdate 2024-10-01 \
            --modelname credit_model_2024_09_01.pkl
        """
    )
    model_monitor_finished = BashOperator(
        task_id="model_monitor_finished",
        bash_command="echo 'Model monitoring completed'"
    )

    model_inference_completed >> model_monitor_start >> model_monitor >> model_monitor_finished

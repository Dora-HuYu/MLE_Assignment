import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
from pyspark.sql import SparkSession
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def read_all_gold_table(table_name: str, gold_root_dir: str, spark: SparkSession):
    folder_path = os.path.join(gold_root_dir, table_name)
    files_list = glob.glob(os.path.join(folder_path, "*.parquet"))

    if not files_list:
        raise FileNotFoundError(f"❌ No parquet files found in {folder_path}")

    print(f"📂 Reading {len(files_list)} files from {folder_path}")
    df = spark.read.option("header", "true").parquet(*files_list)
    return df

def main(snapshotdate: str, modelname: str):
    print('\n\n--- Starting model inference job ---\n')

    spark = SparkSession.builder.appName("ModelInference").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    snapshot_date_str = snapshotdate
    model_name = modelname
    train_test_period_months = 12
    oot_period_months = 2
    train_test_ratio = 0.8

    config = {
        "snapshot_date_str": snapshot_date_str,
        "snapshot_date": datetime.strptime(snapshot_date_str, "%Y-%m-%d"),
        "model_name": model_name,
        "model_bank_directory": "/opt/airflow/model_bank/",
        "model_artefact_filepath": os.path.join("/opt/airflow/model_bank/", model_name),
        "model_train_date_str": snapshot_date_str,  # 以 snapshotdate 作为模型训练时间
        "train_test_period_months": train_test_period_months,
        "oot_period_months": oot_period_months,
        "train_test_ratio": train_test_ratio,
    }

    config["model_train_date"] = datetime.strptime(config["model_train_date_str"], "%Y-%m-%d")
    config["oot_end_date"] = config['model_train_date'] - timedelta(days=1)
    config["oot_start_date"] = config['model_train_date'] - relativedelta(months=oot_period_months)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=train_test_period_months)

    pprint.pprint(config)

    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    print(f"✅ Model loaded: {config['model_artefact_filepath']}")

    X_spark = read_all_gold_table('feature_store', '/opt/airflow/datamart/gold', spark)
    y_spark = read_all_gold_table('label_store', '/opt/airflow/datamart/gold', spark)
    X_df = X_spark.toPandas().sort_values(by='customer_id')
    y_df = y_spark.toPandas().sort_values(by='customer_id')

    X_df['snapshot_date'] = pd.to_datetime(X_df['snapshot_date'])
    y_df['snapshot_date'] = pd.to_datetime(y_df['snapshot_date'])

    y_model_df = y_df[(y_df['snapshot_date'] >= config['train_test_start_date']) & (y_df['snapshot_date'] <= config['model_train_date'])]
    X_model_df = X_df[np.isin(X_df['customer_id'], y_model_df['customer_id'].unique())]

    y_oot = y_model_df[(y_model_df['snapshot_date'] >= config['oot_start_date']) & (y_model_df['snapshot_date'] <= config['oot_end_date'])]
    X_oot = X_model_df[np.isin(X_model_df['customer_id'], y_oot['customer_id'].unique())]

    y_traintest = y_model_df[y_model_df['snapshot_date'] <= config['train_test_end_date']]
    X_traintest = X_model_df[np.isin(X_model_df['customer_id'], y_traintest['customer_id'].unique())]

    X_train, X_test, y_train, y_test = train_test_split(
        X_traintest, y_traintest,
        test_size=config['train_test_ratio'],
        random_state=88,
        shuffle=True,
        stratify=y_traintest['label']
    )

    # drop non-feature columns
    X_test_df = X_test.copy()
    X_test_ordered = X_test_df[model_artefact["feature_columns"]]
    X_test_arr = X_test_ordered.values

    # preprocessing
    transformer = model_artefact['preprocessing_transformers']['stdscaler']
    X_inference = transformer.transform(X_test_arr)

    # predict
    model = model_artefact["model"]
    y_inference = model.predict_proba(X_inference)[:, 1]

    # output result
    y_inference_pdf = X_test_df[["customer_id", "snapshot_date"]].copy()
    y_inference_pdf["model_name"] = model_artefact["model_version"]
    y_inference_pdf["model_predictions"] = y_inference

    gold_directory = f"/opt/airflow/datamart/gold/model_predictions/{model_name[:-4]}"
    os.makedirs(gold_directory, exist_ok=True)

    partition_name = f"{model_name[:-4]}_predictions_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(gold_directory, partition_name)

    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    print(f"✅ Saved inference result to: {filepath}")

    spark.stop()
    print('\n\n--- Completed model inference job ---\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="Model filename (e.g., credit_model_2024_09_01.pkl)")
    args = parser.parse_args()

    main(args.snapshotdate, args.modelname)

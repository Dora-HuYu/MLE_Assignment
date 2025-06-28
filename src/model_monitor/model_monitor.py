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

def calculate_psi(expected_array, actual_array, buckets=10):
    expected = pd.Series(expected_array)
    actual = pd.Series(actual_array)
    bin_edges = np.linspace(0, 1, buckets + 1)
    expected_bins = pd.cut(expected, bins=bin_edges, include_lowest=True)
    actual_bins = pd.cut(actual, bins=bin_edges, include_lowest=True)
    expected_perc = expected_bins.value_counts(normalize=True).sort_index().replace(0, 1e-6)
    actual_perc = actual_bins.value_counts(normalize=True).sort_index().replace(0, 1e-6)
    psi_values = (actual_perc - expected_perc) * np.log(actual_perc / expected_perc)
    psi_df = pd.DataFrame({
        'Bin': expected_perc.index.astype(str),
        'Expected_Percentage': expected_perc.values,
        'Actual_Percentage': actual_perc.values,
        'PSI_Value': psi_values.values
    })
    psi_value = psi_values.sum()
    return psi_df, psi_value

def plot_psi(psi_df, output_path=None):
    psi_df.plot(x='Bin', y=['Expected_Percentage', 'Actual_Percentage'], kind='bar')
    plt.title(f"PSI by Bin (Total PSI={psi_df['PSI_Value'].sum():.4f})")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"✅ PSI plot saved to: {output_path}")
    else:
        plt.show()

def main(snapshotdate_str: str, modelname: str):
    print('\n\n--- Starting model monitor job ---\n')

    # Initialize Spark
    spark = SparkSession.builder.appName("model_monitor").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Setup config
    model_train_date_str = "2024-10-01"
    train_test_months = 12
    oot_months = 2
    train_test_ratio = 0.8

    config = {
        "snapshot_date_str": snapshotdate_str,
        "snapshot_date": datetime.strptime(snapshotdate_str, "%Y-%m-%d"),
        "model_name": modelname,
        "model_bank_directory": "/opt/airflow/model_bank/",
        "model_artefact_filepath": f"/opt/airflow/model_bank/{modelname}",
        "model_train_date_str": model_train_date_str,
        "model_train_date": datetime.strptime(model_train_date_str, "%Y-%m-%d"),
        "oot_period_months": oot_months,
        "train_test_period_months": train_test_months,
        "train_test_ratio": train_test_ratio
    }
    config["oot_end_date"] = config["model_train_date"] - timedelta(days=1)
    config["oot_start_date"] = config["model_train_date"] - relativedelta(months=oot_months)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=train_test_months)

    pprint.pprint(config)

    # Load model
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    model = model_artefact["model"]
    transformer = model_artefact['preprocessing_transformers']['stdscaler']
    feature_columns = model_artefact["feature_columns"]

    print("✅ Model loaded:", config["model_artefact_filepath"])

    # Load data
    X_spark = read_all_gold_table('feature_store', '/opt/airflow/datamart/gold', spark)
    y_spark = read_all_gold_table('label_store', '/opt/airflow/datamart/gold', spark)

    X_df = X_spark.toPandas().sort_values('customer_id')
    y_df = y_spark.toPandas().sort_values('customer_id')
    X_df['snapshot_date'] = pd.to_datetime(X_df['snapshot_date'])
    y_df['snapshot_date'] = pd.to_datetime(y_df['snapshot_date'])

    y_model_df = y_df[(y_df['snapshot_date'] >= config['train_test_start_date']) & (y_df['snapshot_date'] <= config['model_train_date'])]
    X_model_df = X_df[np.isin(X_df['customer_id'], y_model_df['customer_id'])]

    # Split
    y_oot = y_model_df[(y_model_df['snapshot_date'] >= config['oot_start_date']) & (y_model_df['snapshot_date'] <= config['oot_end_date'])]
    X_oot = X_model_df[np.isin(X_model_df['customer_id'], y_oot['customer_id'])]

    y_traintest = y_model_df[y_model_df['snapshot_date'] <= config['train_test_end_date']]
    X_traintest = X_model_df[np.isin(X_model_df['customer_id'], y_traintest['customer_id'])]

    X_train, X_test, y_train, y_test = train_test_split(
        X_traintest, y_traintest, test_size=config['train_test_ratio'], random_state=88, stratify=y_traintest['label']
    )

    # Preprocess
    X_test_arr = transformer.transform(X_test[feature_columns])
    X_oot_arr = transformer.transform(X_oot[feature_columns])
    y_oot_arr = y_oot['label'].values
    y_train_arr = y_train['label'].values

    # Predict
    y_monitor = model.predict_proba(X_test_arr)[:, 1]
    monitor_df = X_test[['customer_id', 'snapshot_date']].copy()
    monitor_df['model_name'] = modelname
    monitor_df['model_predictions'] = y_monitor

    # AUC metrics
    test_auc = roc_auc_score(y_test['label'], y_monitor)
    oot_auc = roc_auc_score(y_oot_arr, model.predict_proba(X_oot_arr)[:, 1])
    print(f"✅ Test AUC: {test_auc:.4f}, GINI: {2 * test_auc - 1:.4f}")
    print(f"✅ OOT AUC:  {oot_auc:.4f}, GINI: {2 * oot_auc - 1:.4f}")

    # PSI
    psi_df, psi_val = calculate_psi(y_train_arr, y_monitor)
    print(f"📊 Total PSI: {psi_val:.4f}")
    plot_psi(psi_df, output_path=f"/opt/airflow/output/psi_plot_{snapshotdate_str}.png")

    # Save results
    gold_dir = f"/opt/airflow/datamart/gold/model_monitors/{modelname[:-4]}/"
    os.makedirs(gold_dir, exist_ok=True)
    output_file = os.path.join(gold_dir, f"{modelname[:-4]}_monitor_{snapshotdate_str.replace('-', '_')}.parquet")
    spark.createDataFrame(monitor_df).write.mode("overwrite").parquet(output_file)
    print(f"✅ Monitor data saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshotdate', type=str, required=True)
    parser.add_argument('--modelname', type=str, required=True)
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)

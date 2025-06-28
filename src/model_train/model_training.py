import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from pyspark.sql import SparkSession


# to call this script: python model_train.py --snapshotdate "2024-09-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # --- set up config ---
    model_train_date_str = "2024-09-01"
    train_test_period_months = 12
    oot_period_months = 2
    train_test_ratio = 0.8

    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] =  oot_period_months
    config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
    config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
    config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
    config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
    config["train_test_ratio"] = train_test_ratio 
    pprint.pprint(config)


    # --- Import data from gold table---
    def read_all_gold_table(table_name: str, gold_root_dir: str, spark: SparkSession):
        folder_path = os.path.join(gold_root_dir, table_name)
        subdirs = glob.glob(os.path.join(folder_path, "20*.parquet"))  # 匹配所有形如 2023_01_01.parquet 的子目录

        if not subdirs:
            raise FileNotFoundError(f"No subdirectories found in {folder_path}")
    
        print(f"🔍 Reading from {len(subdirs)} snapshot folders under {folder_path}")
    
        df = spark.read.parquet(*subdirs)
        return df

    X_spark = read_all_gold_table('feature_store', '/opt/airflow/datamart/gold', spark)
    y_spark = read_all_gold_table('label_store', '/opt/airflow/datamart/gold', spark)

    X_df = X_spark.toPandas().sort_values(by='customer_id')
    y_df = y_spark.toPandas().sort_values(by='customer_id')

    # --- Set train test split ---
    # Consider data from model training date
    # Makesure snapshot_date tyoes are the same
    y_df['snapshot_date'] = pd.to_datetime(y_df['snapshot_date'])
    X_df['snapshot_date'] = pd.to_datetime(X_df['snapshot_date'])

    y_model_df = y_df[(y_df['snapshot_date'] >= config['train_test_start_date']) & (y_df['snapshot_date'] <= config['model_train_date'])]
    X_model_df = X_df[np.isin(X_df['customer_id'], y_model_df['customer_id'].unique())]

    # Create OOT split
    y_oot = y_model_df[(y_model_df['snapshot_date'] >= config['oot_start_date']) & (y_model_df['snapshot_date'] <= config['oot_end_date'])]
    X_oot = X_model_df[np.isin(X_model_df['customer_id'], y_oot['customer_id'].unique())]

    # Everything else goes into train-test
    y_traintest = y_model_df[y_model_df['snapshot_date'] <= config['train_test_end_date']]
    X_traintest = X_model_df[np.isin(X_model_df['customer_id'], y_traintest['customer_id'].unique())]


    X_train, X_test, y_train, y_test = train_test_split(X_traintest, y_traintest, 
                                                    test_size=config['train_test_ratio'], 
                                                    random_state=88, 
                                                    shuffle=True, 
                                                    stratify=y_traintest['label'])

    print('X_train', X_train.shape[0])
    print('X_test', X_test.shape[0])
    print('X_oot', X_oot.shape[0])
    print('y_train', y_train.shape[0], round(y_train['label'].mean(), 2))
    print('y_test', y_test.shape[0], round(y_test['label'].mean(), 2))
    print('y_oot', y_oot.shape[0], round(y_oot['label'].mean(), 2))

    X_train

    # --- Process Data ---
    # Transform data into numpy arrays
    X_train_arr = X_train.drop(columns=['customer_id', 'snapshot_date']).values
    X_test_arr = X_test.drop(columns=['customer_id', 'snapshot_date']).values
    X_oot_arr = X_oot.drop(columns=['customer_id', 'snapshot_date']).values

    y_train_arr = y_train['label'].values
    y_test_arr = y_test['label'].values
    y_oot_arr = y_oot['label'].values

    scaler = StandardScaler()
    transformer_stdscaler = scaler.fit(X_train_arr)

    X_train_arr = transformer_stdscaler.fit_transform(X_train_arr)
    X_test_arr = transformer_stdscaler.transform(X_test_arr)
    X_oot_arr = transformer_stdscaler.transform(X_oot_arr)

    pd.DataFrame(X_train_arr)

    # --- train model ---
    #a. Logistic Regression
    # Train model
    clf = LogisticRegression()
    clf.fit(X_train_arr, y_train_arr)

    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
        
    # --- Test load pickle and make model inference ---
    # Load model
    with open("model.pkl", "rb") as f:
        clf = pickle.load(f)
    # Predict and evaluate
    y_pred_proba_train = clf.predict_proba(X_train_arr)[:, 1]
    train_auc = roc_auc_score(y_train_arr, y_pred_proba_train)

    y_pred_proba_test = clf.predict_proba(X_test_arr)[:, 1]
    test_auc = roc_auc_score(y_test_arr, y_pred_proba_test)

    y_pred_proba_oot = clf.predict_proba(X_oot_arr)[:, 1]
    oot_auc = roc_auc_score(y_oot_arr, y_pred_proba_oot)

    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"OOT AUC: {oot_auc:.4f}")

    # F2 score across thresholds
    thresholds = np.arange(0.0, 1.0, 0.01)
    beta = 1.5
    f1_scores_train = [fbeta_score(y_train_arr, y_pred_proba_train > t, beta=beta) for t in thresholds]
    f1_scores_test = [fbeta_score(y_test_arr, y_pred_proba_test > t, beta=beta) for t in thresholds]
    f1_scores_oot = [fbeta_score(y_oot_arr, y_pred_proba_oot > t, beta=beta) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores_train)]

    # Plot F1 Score vs. Threshold
    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, f1_scores_train, label=f"Train F-{beta} Score")
    plt.plot(thresholds, f1_scores_test, label=f"Test F-{beta} Score")
    plt.plot(thresholds, f1_scores_oot, label=f"OOT F-{beta} Score")
    plt.axvline(x=best_threshold, color="red", linestyle="--", label=f"Best Threshold: {best_threshold:.2f}")
    plt.title(f"F-{beta} Score vs. Probability Threshold")
    plt.xlabel("Threshold")
    plt.ylabel(f"F-{beta} Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Predictions
    y_pred_train = (y_pred_proba_train > best_threshold).astype(int)
    y_pred_test = (y_pred_proba_test > best_threshold).astype(int)
    y_pred_oot = (y_pred_proba_oot > best_threshold).astype(int)

    # Confusion matrices
    cm_train = confusion_matrix(y_train_arr, y_pred_train)
    cm_test = confusion_matrix(y_test_arr, y_pred_test)
    cm_oot = confusion_matrix(y_oot_arr, y_pred_oot)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Train
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    disp_train.plot(ax=axs[0], cmap='Blues', colorbar=False)
    axs[0].set_title(f"Train (Threshold={best_threshold:.2f})")
    axs[0].grid(False)

    # Test
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    disp_test.plot(ax=axs[1], cmap='Blues', colorbar=False)
    axs[1].set_title(f"Test (Threshold={best_threshold:.2f})")
    axs[1].grid(False)

    # OOT
    disp_oot = ConfusionMatrixDisplay(confusion_matrix=cm_oot)
    disp_oot.plot(ax=axs[2], cmap='Blues', colorbar=False)
    axs[2].set_title(f"OOT (Threshold={best_threshold:.2f})")
    axs[2].grid(False)

    plt.tight_layout()
    plt.show()

    print(f"Best Train F{beta}-Score: {max(f1_scores_train):.4f}")
    print(f"Best Test F{beta}-Score: {max(f1_scores_test):.4f}")
    print(f"Best OOT F{beta}-Score: {max(f1_scores_oot):.4f}")

    #b. XGBoost
    # Define the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=88)

    # Define the hyperparameter space to search
    param_dist = {
        'n_estimators': [25, 50],
        'max_depth': [2, 3],  # lower max_depth to simplify the model
        'learning_rate': [0.01, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }

    # Create a scorer based on AUC score
    auc_scorer = make_scorer(roc_auc_score)

    # Set up the random search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        scoring=auc_scorer,
        n_iter=100,  # Number of iterations for random search
        cv=3,       # Number of folds in cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1   # Use all available cores
    )

    # Perform the random search
    random_search.fit(X_train_arr, y_train_arr)

    # Output the best parameters and best score
    print("Best parameters found: ", random_search.best_params_)
    print("Best AUC score: ", random_search.best_score_)

    # Evaluate the model on the train set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_train_arr)[:, 1]
    train_auc_score = roc_auc_score(y_train_arr, y_pred_proba)
    print("Train AUC score: ", train_auc_score)

    # Evaluate the model on the test set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_arr)[:, 1]
    test_auc_score = roc_auc_score(y_test_arr, y_pred_proba)
    print("Test AUC score: ", test_auc_score)

    # Evaluate the model on the oot set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_oot_arr)[:, 1]
    oot_auc_score = roc_auc_score(y_oot_arr, y_pred_proba)
    print("OOT AUC score: ", oot_auc_score)

    print("TRAIN GINI score: ", round(2*train_auc_score-1,3))
    print("Test GINI score: ", round(2*test_auc_score-1,3))
    print("OOT GINI score: ", round(2*oot_auc_score-1,3))

    # F2 score across thresholds
    thresholds = np.arange(0.0, 1.0, 0.01)
    beta = 1.5
    f2_scores_train = [fbeta_score(y_train_arr, y_pred_proba_train > t, beta=beta) for t in thresholds]
    f2_scores_test = [fbeta_score(y_test_arr, y_pred_proba_test > t, beta=beta) for t in thresholds]
    f2_scores_oot = [fbeta_score(y_oot_arr, y_pred_proba_oot > t, beta=beta) for t in thresholds]

    #Find the best trshhold on F2
    best_threshold = thresholds[np.argmax(f2_scores_train)]

    # Plotting F1.5 vs Threshold
    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, f2_scores_train, label=f"Train F-{beta} Score")
    plt.plot(thresholds, f2_scores_test, label=f"Test F-{beta} Score")
    plt.plot(thresholds, f2_scores_oot, label=f"OOT F-{beta} Score")
    plt.axvline(x=best_threshold, color="red", linestyle="--", label=f"Best Threshold: {best_threshold:.2f}")
    plt.title(f"XGBoost F-{beta} Score vs. Probability Threshold")
    plt.xlabel("Threshold")
    plt.ylabel(f"F-{beta} Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print Best Threshold
    print(f"Best threshold (Train F-{beta} max): {best_threshold:.2f}")
    # Predictions
    y_pred_train = (y_pred_proba_train > best_threshold).astype(int)
    y_pred_test = (y_pred_proba_test > best_threshold).astype(int)
    y_pred_oot = (y_pred_proba_oot > best_threshold).astype(int)

    # Confusion matrices
    cm_train = confusion_matrix(y_train_arr, y_pred_train)
    cm_test = confusion_matrix(y_test_arr, y_pred_test)
    cm_oot = confusion_matrix(y_oot_arr, y_pred_oot)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Train
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    disp_train.plot(ax=axs[0], cmap='Blues', colorbar=False)
    axs[0].set_title(f"Train (Threshold={best_threshold:.2f})")
    axs[0].grid(False)

    # Test
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    disp_test.plot(ax=axs[1], cmap='Blues', colorbar=False)
    axs[1].set_title(f"Test (Threshold={best_threshold:.2f})")
    axs[1].grid(False)

    # OOT
    disp_oot = ConfusionMatrixDisplay(confusion_matrix=cm_oot)
    disp_oot.plot(ax=axs[2], cmap='Blues', colorbar=False)
    axs[2].set_title(f"OOT (Threshold={best_threshold:.2f})")
    axs[2].grid(False)

    plt.tight_layout()
    plt.show()

    # default 0.5
    print("\n👉 Default threshold 0.5 results:")
    precision_05_train = precision_score(y_train_arr, y_pred_proba_train > 0.5)
    recall_05_train = recall_score(y_train_arr, y_pred_proba_train > 0.5)
    f2_05_train = fbeta_score(y_train_arr, y_pred_proba_train > 0.5, beta=1.5)

    precision_05_test = precision_score(y_test_arr, y_pred_proba_test > 0.5)
    recall_05_test = recall_score(y_test_arr, y_pred_proba_test > 0.5)
    f2_05_test = fbeta_score(y_test_arr, y_pred_proba_test > 0.5, beta=1.5)

    precision_05_oot = precision_score(y_oot_arr, y_pred_proba_oot > 0.5)
    recall_05_oot = recall_score(y_oot_arr, y_pred_proba_oot > 0.5)
    f2_05_oot = fbeta_score(y_oot_arr, y_pred_proba_oot > 0.5, beta=1.5)

    print(f"Train: Precision={precision_05_train:.3f}, Recall={recall_05_train:.3f}, F1.5={f2_05_train:.3f}")
    print(f"Test: Precision={precision_05_test:.3f}, Recall={recall_05_test:.3f}, F1.5={f2_05_test:.3f}")
    print(f"OOT: Precision={precision_05_oot:.3f}, Recall={recall_05_oot:.3f}, F1.5={f2_05_oot:.3f}")

    # best threshold from your F2 analysis (0.27)
    print("\n👉 Best threshold 0.27 results:")
    precision_best_train = precision_score(y_train_arr, y_pred_proba_train > 0.27)
    recall_best_train = recall_score(y_train_arr, y_pred_proba_train > 0.27)
    f2_best_train = fbeta_score(y_train_arr, y_pred_proba_train > 0.27, beta=1.5)

    precision_best_test = precision_score(y_test_arr, y_pred_proba_test > 0.27)
    recall_best_test = recall_score(y_test_arr, y_pred_proba_test > 0.27)
    f2_best_test = fbeta_score(y_test_arr, y_pred_proba_test > 0.27, beta=1.5)

    precision_best_oot = precision_score(y_oot_arr, y_pred_proba_oot > 0.27)
    recall_best_oot = recall_score(y_oot_arr, y_pred_proba_oot > 0.27)
    f2_best_oot = fbeta_score(y_oot_arr, y_pred_proba_oot > 0.27, beta=1.5)

    print(f"Train: Precision={precision_best_train:.3f}, Recall={recall_best_train:.3f}, F1.5={f2_best_train:.3f}")
    print(f"Test: Precision={precision_best_test:.3f}, Recall={recall_best_test:.3f}, F1.5={f2_best_test:.3f}")
    print(f"OOT: Precision={precision_best_oot:.3f}, Recall={recall_best_oot:.3f}, F1.5={f2_best_oot:.3f}")

    # --- Prepare model artefact to save ---
    model_artefact = {}

    model_artefact['model'] = best_model
    model_artefact['model_version'] = "credit_model_"+config["model_train_date_str"].replace('-','_')
    model_artefact['preprocessing_transformers'] = {}
    model_artefact['preprocessing_transformers']['stdscaler'] = transformer_stdscaler
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {}
    model_artefact['data_stats']['X_train_arr'] = X_train_arr.shape[0]
    model_artefact['data_stats']['X_test_arr'] = X_test_arr.shape[0]
    model_artefact['data_stats']['X_oot_arr'] = X_oot_arr.shape[0]
    model_artefact['data_stats']['y_train_arr'] = round(y_train_arr.mean(),2)
    model_artefact['data_stats']['y_test_arr'] = round(y_test_arr.mean(),2)
    model_artefact['data_stats']['y_oot_arr'] = round(y_oot_arr.mean(),2)
    model_artefact['results'] = {}
    model_artefact['results']['auc_train'] = train_auc_score
    model_artefact['results']['auc_test'] = test_auc_score
    model_artefact['results']['auc_oot'] = oot_auc_score
    model_artefact['results']['gini_train'] = round(2*train_auc_score-1,3)
    model_artefact['results']['gini_test'] = round(2*test_auc_score-1,3)
    model_artefact['results']['gini_oot'] = round(2*oot_auc_score-1,3)
    model_artefact['hp_params'] = random_search.best_params_
    pprint.pprint(model_artefact)

    # --- save artefact to model bank ---
    # create model_bank dir
    model_bank_directory = "model_bank/"

    if not os.path.exists(model_bank_directory):
        os.makedirs(model_bank_directory)

    # Full path to the file
    file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')

    # Write the model to a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)

    print(f"Model saved to {file_path}")

    # --- test load pickle and make model inference ---
    # Load the model from the pickle file
    with open(file_path, 'rb') as file:
        loaded_model_artefact = pickle.load(file)

    y_pred_proba = loaded_model_artefact['model'].predict_proba(X_oot_arr)[:, 1]
    oot_auc_score = roc_auc_score(y_oot_arr, y_pred_proba)
    print("OOT AUC score: ", oot_auc_score)

    print("Model loaded successfully!")

    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')



if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)













    

    


    
import os
import json

def check_data_quality():
    file_list = [
        "/opt/airflow/data/feature_clickstream.csv",
        "/opt/airflow/data/features_attributes.csv",
        "/opt/airflow/data/features_financials.csv",
        "/opt/airflow/data/lms_loan_daily.csv",
    ]

    output_dir = "/opt/airflow/outputs/check_data_quality"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for file_path in file_list:
        status = {"file": file_path}
        if not os.path.exists(file_path):
            status["status"] = "missing"
        elif os.path.getsize(file_path) == 0:
            status["status"] = "empty"
        else:
            status["status"] = "ready"
        print(f"{status['status'].upper()} - {file_path}")
        results.append(status)

    # 写入检查结果
    with open(os.path.join(output_dir, "check_data_ready_result.json"), "w") as f:
        json.dump(results, f, indent=4)

    # 如果有任何失败就抛异常
    failures = [r for r in results if r["status"] != "ready"]
    if failures:
        raise ValueError(f"❌ Files failed check: {[f['file'] for f in failures]}")
 
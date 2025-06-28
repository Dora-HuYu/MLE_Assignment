import os
import json

def check_csv_file(path):
    """
    检查 CSV 文件是否存在且不为空
    """
    if not os.path.exists(path):
        return {"file": path, "status": "missing"}
    if os.path.getsize(path) == 0:
        return {"file": path, "status": "empty"}
    return {"file": path, "status": "ready"}

def check_source_data_bronze_label():
    files_to_check = [
        "/opt/airflow/data/lms_loan_daily.csv",
    ]

    output_dir = "/opt/airflow/outputs/bronze_label_check"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for file_path in files_to_check:
        result = check_csv_file(file_path) 
        print(f"{result['status'].upper()} - {result['file']}")
        results.append(result)

    output_file = os.path.join(output_dir, "check_label_ready_result.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    failures = [r for r in results if r["status"] != "ready"]
    if failures:
        failed_files = "\n".join(f["file"] for f in failures)
        raise RuntimeError(f"❌ Data check failed for the following files:\n{failed_files}")

    print(f"✅ All CSV files passed check. Results saved to {output_file}")

import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType

def process_gold_table(
    snapshot_date_str,
    silver_loan_daily_directory,
    silver_attr_directory,
    silver_fin_directory,
    gold_label_store_directory,
    spark,
    mob=3,
    dpd=30
):
    # Step 1: Prepare snapshot_date
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Step 2: Load silver tables (⚠️ 不再 lowercase)
    loan_path = os.path.join(silver_loan_daily_directory, f"silver_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet")
    attr_path = os.path.join(silver_attr_directory, f"silver_attributes_{snapshot_date_str.replace('-', '_')}.parquet")
    fin_path  = os.path.join(silver_fin_directory, f"silver_financials_{snapshot_date_str.replace('-', '_')}.parquet")

    df_loan = spark.read.parquet(loan_path)
    df_attr = spark.read.parquet(attr_path)
    df_fin  = spark.read.parquet(fin_path)

    print("Loaded loan:", df_loan.count(), "attr:", df_attr.count(), "fin:", df_fin.count())

    # Step 3: Filter by MOB
    df_loan = df_loan.filter(F.col("mob") == mob)

    # Step 4: Join on Customer_ID (⚠️ 大小写正确)
    df_gold = df_loan.alias("loan") \
        .join(df_attr.alias("attr"), on="Customer_ID", how="left") \
        .join(df_fin.alias("fin"), on="Customer_ID", how="left")

    # Step 5: Add label
    df_gold = df_gold.withColumn("label", F.when(F.col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df_gold = df_gold.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))

    # Step 6: Select columns (⚠️ 保持原始字段名大小写)
    df_gold = df_gold.select(
        col("loan.loan_id"),
        col("loan.Customer_ID"),
        col("label"),
        col("label_def"),
        col("loan.snapshot_date"),
        col("fin.Annual_Income"),
        col("fin.Num_Credit_Card"),
        col("fin.Credit_Utilization_Ratio"),
        col("fin.debt_to_income_ratio")
    )

    # Step 7: Save
    output_path = os.path.join(gold_label_store_directory, f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet")
    df_gold.write.mode("overwrite").parquet(output_path)

    print("✅ Saved to:", output_path)
    return df_gold

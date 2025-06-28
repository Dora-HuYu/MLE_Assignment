
import os
import glob
import pyspark
import pyspark.sql.functions as F

from tqdm import tqdm

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, MapType, NumericType, ArrayType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler, StandardScaler

def read_silver_table(table, silver_db, spark):
    """
    Helper function to read all partitions of a silver table
    """
    folder_path = os.path.join(silver_db, table)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").parquet(*files_list)
    return df

############################
# Label Store
############################
def build_label_store(mob, dpd, df):
    """
    Function to build label store
    """
    ####################
    # Create labels
    ####################

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "customer_id", "label", "label_def", "snapshot_date")

    return df

############################
# Pipeline
############################

def process_gold_table(silver_db, gold_db, partitions_list, spark):
    """
    Wrapper function to build all gold tables
    """
    # Read silver tables
    df_attributes = read_silver_table('attributes', silver_db, spark)
    df_clickstream = read_silver_table('clickstream', silver_db, spark)
    df_financials = read_silver_table('financials', silver_db, spark)
    df_loan_type = read_silver_table('loan_type', silver_db, spark)
    df_lms = read_silver_table('lms', silver_db, spark)

    # Build label store
    print("Building label store...")
    df_label = build_label_store(6, 30, df_lms)
    
    # Build features
    print("Building features...")
    df_features = build_feature_store(df_attributes, df_financials, df_loan_type, df_clickstream, df_lms, df_label)

    # Partition and save features
    for date_str in tqdm(partitions_list, total=len(partitions_list), desc="Saving features"):
        partition_name = date_str.replace('-','_') + '.parquet'
        feature_filepath = os.path.join(gold_db, 'feature_store', partition_name)
        df_features.filter(col('snapshot_date')==date_str).write.mode('overwrite').parquet(feature_filepath)

    # Partition and save labels
    for date_str in tqdm(partitions_list, total=len(partitions_list), desc="Saving labels"):
        partition_name = date_str.replace('-','_') + '.parquet'
        label_filepath = os.path.join(gold_db, 'label_store', partition_name)
        df_label.filter(col('snapshot_date')==date_str).write.mode('overwrite').parquet(label_filepath)

    return df_features, df_label
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    attr_path = "data/features_attributes.csv"
    fin_path = "data/features_financials.csv"
    loan_path = "data/lms_loan_daily.csv"

    # load data - IRL ingest from back end source system
    df_attr = spark.read.csv(attr_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df_attr.count())
    df_fin = spark.read.csv(fin_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df_fin.count())
    df_loan = spark.read.csv(loan_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df_loan.count())

    
    # save bronze attributes table to datamart - IRL connect to database to write
    attr_filename = "bronze_attributes_" + snapshot_date_str.replace('-', '_') + ".csv"
    attr_filepath = os.path.join(bronze_lms_directory, attr_filename)
    df_attr.toPandas().to_csv(attr_filepath, index=False)
    print("Saved attributes to:", attr_filepath)

    # save bronze financials table to datamart - IRL connect to database to write
    fin_filename = "bronze_financials_" + snapshot_date_str.replace('-', '_') + ".csv"
    fin_filepath = os.path.join(bronze_lms_directory, fin_filename)
    df_fin.toPandas().to_csv(fin_filepath, index=False)
    print("Saved financials to:", fin_filepath)

    # save loan table to datamart
    loan_filename = "bronze_loan_daily_" + snapshot_date_str.replace('-', '_') + ".csv"
    loan_filepath = os.path.join(bronze_lms_directory, loan_filename)
    df_loan.toPandas().to_csv(loan_filepath, index=False)
    print("Saved loan to:", loan_filepath)

    # return all three DataFrames
    return df_attr, df_fin, df_loan

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
from pyspark.sql.functions import lit

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.functions import ceil, when, datediff, split, expr, regexp_replace, trim

def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, silver_attr_directory, silver_fin_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

   # Load bronze tables
    partition_name_loan = "bronze_loan_daily_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath_loan = bronze_lms_directory + partition_name_loan
    df_loan = spark.read.csv(filepath_loan, header=True, inferSchema=True).withColumn("snapshot_date", lit(snapshot_date).cast(DateType()))
    print('Loaded loan_daily from:', filepath_loan, 'Row count:', df_loan.count())

    partition_name_attr = "bronze_attributes_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath_attr = bronze_lms_directory + partition_name_attr
    df_attr = spark.read.csv(filepath_attr, header=True, inferSchema=True).withColumn("snapshot_date", lit(snapshot_date).cast(DateType()))
    print('Loaded attributes from:', filepath_attr, 'Row count:', df_attr.count())

    partition_name_fin = "bronze_financials_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath_fin = bronze_lms_directory + partition_name_fin
    df_fin = spark.read.csv(filepath_fin, header=True, inferSchema=True).withColumn("snapshot_date", lit(snapshot_date).cast(DateType()))
    print('Loaded financials from:', filepath_fin, 'Row count:', df_fin.count())


    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    # ==== Loan_Daily ====
    column_type_map_loan = {
    "loan_id": StringType(),
    "Customer_ID": StringType(),
    "loan_start_date": DateType(),
    "tenure": IntegerType(),
    "installment_num": IntegerType(),
    "loan_amt": FloatType(),
    "due_amt": FloatType(),
    "paid_amt": FloatType(),
    "overdue_amt": FloatType(),
    "balance": FloatType(),
    "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map_loan.items():
        df_loan = df_loan.withColumn(column, col(column).cast(new_type))
    

    # ==== Attributes ====
    column_type_map_attr = {
    "Customer_ID": StringType(),
    "Name": StringType(),
    "Age": IntegerType(),
    "SSN": StringType(),
    "Occupation": StringType(),
    "snapshot_date": DateType(),
    }
    
    for column, new_type in column_type_map_attr.items():
        df_attr = df_attr.withColumn(column, col(column).cast(new_type))
    

    # ==== Finance ====
    column_type_map_fin = {
    "Customer_ID": StringType(),
    "Annual_Income": FloatType(),
    "Monthly_Inhand_Salary": FloatType(),
    "Num_Bank_Accounts": IntegerType(),
    "Num_Credit_Card": IntegerType(),
    "Interest_Rate": IntegerType(),
    "Num_of_Loan": IntegerType(),
    "Type_of_Loan": StringType(),
    "Delay_from_due_date": IntegerType(),
    "Num_of_Delayed_Payment": IntegerType(),
    "Changed_Credit_Limit": FloatType(),
    "Num_Credit_Inquiries": FloatType(),
    "Credit_Mix": StringType(),
    "Outstanding_Debt": FloatType(),
    "Credit_Utilization_Ratio": FloatType(),
    "Credit_History_Age": StringType(),  # 这个需要后处理拆解成年+月
    "Payment_of_Min_Amount": StringType(),
    "Total_EMI_per_month": FloatType(),
    "Amount_invested_monthly": FloatType(),
    "Payment_Behaviour": StringType(),
    "Monthly_Balance": FloatType(),
    "snapshot_date": DateType()
    }
    for column, new_type in column_type_map_fin.items():
        df_fin = df_fin.withColumn(column, col(column).cast(new_type))
    

    # ==== Loan_Daily ====
    # augment data: add month on book (MOB)
    df_loan = df_loan.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: estimate how many installments are missed
    df_loan = df_loan.withColumn("installments_missed",ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    # derive the first missed date
    df_loan = df_loan.withColumn("first_missed_date",when(col("installments_missed") > 0,F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))


    # compute DPD (days past due)
    df_loan = df_loan.withColumn("dpd",when(col("overdue_amt") > 0.0,datediff(col("snapshot_date"),col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df_loan.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    #return df_loan

    # ===== Attributes =====
    # ===== Clean 'Age' column =====
    # Remove underscores and cast to integer
    df_attr = df_attr.withColumn("Age", regexp_replace("Age", "_", ""))
    df_attr = df_attr.withColumn("Age", col("Age").cast(IntegerType()))
    # Set invalid ages (outside 18–99) to null
    df_attr = df_attr.withColumn("Age",when((col("Age") >= 18) & (col("Age") <= 99), col("Age")).otherwise(None))
    
    # ===== Clean 'Occupation' column =====
    # Trim whitespaces and replace "_______" with null
    df_attr = df_attr.withColumn("Occupation", trim(col("Occupation")))
    df_attr = df_attr.withColumn("Occupation", when(col("Occupation") == "_______", None).otherwise(col("Occupation")))

    # ===== Drop SSN column (garbage values) =====
    df_attr = df_attr.drop("SSN")

    # ===== Derive 'age_group' column =====
    df_attr = df_attr.withColumn("age_group",when(col("Age") < 26, "18-25").when(col("Age") < 36, "26-35").when(col("Age") < 46, "36-45").otherwise("46+"))

    # ===== Save to silver layer (parquet) =====
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-', '_') + ".parquet"
    filepath_attr = silver_attr_directory + partition_name
    df_attr.write.mode("overwrite").parquet(filepath_attr)

    print("Saved to:", filepath_attr)
    #return df_attr

    # ===== Financial =====
    # ===== Clean Annual_Income =====
    df_fin = df_fin.withColumn("Annual_Income", regexp_replace("Annual_Income", "_", ""))
    df_fin = df_fin.withColumn("Annual_Income", col("Annual_Income").cast(FloatType()))
    df_fin = df_fin.withColumn("Annual_Income", expr("ROUND(Annual_Income, 2)"))

    # ===== Clean Monthly_Inhand_Salary =====
    df_fin = df_fin.withColumn("Monthly_Inhand_Salary", expr("ROUND(Monthly_Inhand_Salary, 2)"))

    # ===== Clean Num_Bank_Accounts =====
    df_fin = df_fin.withColumn("Num_Bank_Accounts",when((col("Num_Bank_Accounts") >= 0) & (col("Num_Bank_Accounts") <= 30), col("Num_Bank_Accounts")).otherwise(None))

    # ===== Clean Num_Credit_Card =====
    df_fin = df_fin.withColumn("Num_Credit_Card",when((col("Num_Credit_Card") >= 0) & (col("Num_Credit_Card") <= 35), col("Num_Credit_Card")).otherwise(None))

    # ===== Clean Interest_Rate (convert from % to decimal) =====
    # Remove invalid interest rates
    df_fin = df_fin.withColumn("Interest_Rate", (col("Interest_Rate") / 100.0).cast(FloatType()))
    df_fin = df_fin.withColumn("Interest_Rate",when((col("Interest_Rate") >= 0.01) & (col("Interest_Rate") <= 0.5), col("Interest_Rate")).otherwise(None))

    # ===== Clean Num_of_Loan =====
    df_fin = df_fin.withColumn("Num_of_Loan", regexp_replace("Num_of_Loan", "_", ""))
    df_fin = df_fin.withColumn("Num_of_Loan", col("Num_of_Loan").cast(IntegerType()))
    df_fin = df_fin.withColumn("Num_of_Loan",when((col("Num_of_Loan") >= 0) & (col("Num_of_Loan") <= 15), col("Num_of_Loan")).otherwise(None))

    # ===== Clean Type_of_Loan =====
    # Replace "Not Specified" or blank with null
    df_fin = df_fin.withColumn("Type_of_Loan",when((col("Type_of_Loan").isNull()) | (col("Type_of_Loan") == "Not Specified"), None).otherwise(col("Type_of_Loan")))

    # ===== Clean Delay_from_due_date =====
    df_fin = df_fin.withColumn("Delay_from_due_date",when(col("Delay_from_due_date") >= 0, col("Delay_from_due_date")).otherwise(None))

    # ===== Clean Num_of_Delayed_Payment =====
    df_fin = df_fin.withColumn("Num_of_Delayed_Payment", regexp_replace("Num_of_Delayed_Payment", "_", ""))
    df_fin = df_fin.withColumn("Num_of_Delayed_Payment", col("Num_of_Delayed_Payment").cast(IntegerType()))
    df_fin = df_fin.withColumn("Num_of_Delayed_Payment",when(col("Num_of_Delayed_Payment") >= 0, col("Num_of_Delayed_Payment")).otherwise(None))

    # ===== Clean Credit_Limit =====
    # Remove underscores and convert to float
    df_fin = df_fin.withColumn("Changed_Credit_Limit", regexp_replace("Changed_Credit_Limit", "_", ""))
    df_fin = df_fin.withColumn("Changed_Credit_Limit", col("Changed_Credit_Limit").cast(FloatType()))
    # Set negative values to null
    df_fin = df_fin.withColumn("Changed_Credit_Limit",when(col("Changed_Credit_Limit") >= 0, col("Changed_Credit_Limit")).otherwise(None))
    #Add column “credit_limit_increased” to see if credit increased
    df_fin = df_fin.withColumn("credit_limit_increased", when(col("Changed_Credit_Limit") > 0, 1).otherwise(0))

    # ===== Clean Credit_Mix =====
    df_fin = df_fin.withColumn("Credit_Mix", regexp_replace("Credit_Mix", "_", "None"))
    df_fin = df_fin.withColumn("Credit_Mix", col("Credit_Mix").cast(StringType()))

    # ===== Clean Outstanding_Debt =====
    df_fin = df_fin.withColumn("Outstanding_Debt", regexp_replace("Outstanding_Debt", "_", ""))
    df_fin = df_fin.withColumn("Outstanding_Debt", col("Outstanding_Debt").cast(FloatType()))

    # ===== Credit_History_Age to credit_history_months =====
    df_fin = df_fin.withColumn("years", split(col("Credit_History_Age"), " ").getItem(0).cast(IntegerType()))
    df_fin = df_fin.withColumn("months", split(col("Credit_History_Age"), " ").getItem(3).cast(IntegerType()))
    df_fin = df_fin.withColumn("credit_history_months", (col("years") * 12 + col("months")).cast(IntegerType()))
    df_fin = df_fin.drop("Credit_History_Age", "years", "months")

    # ===== Clean Payment_Behaviour =====
    df_fin = df_fin.withColumn("Payment_Behaviour", regexp_replace("Payment_Behaviour", "!@9#%8", "None"))
    df_fin = df_fin.withColumn("Payment_Behaviour", col("Payment_Behaviour").cast(StringType()))

    # ===== Debt-to-Income Ratio =====
    df_fin = df_fin.withColumn("debt_to_income_ratio",(col("Outstanding_Debt").cast(FloatType()) / col("Annual_Income")).cast(FloatType()))

    # ===== Save to silver layer =====
    partition_name = "silver_financials_" + snapshot_date_str.replace('-', '_') + ".parquet"
    filepath_fin = silver_fin_directory + partition_name
    df_fin.write.mode("overwrite").parquet(filepath_fin)

    print("Saved to:", filepath_fin)


    return df_loan, df_attr, df_fin



    

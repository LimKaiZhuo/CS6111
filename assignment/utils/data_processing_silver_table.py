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

from pyspark.sql.functions import col, when, trim, regexp_replace, avg, round as spark_round,  mean as spark_mean
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
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

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_table_feature_attribute(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    fa = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', fa.count())
    ###################
    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    # enfore datatype
    for column, new_type in column_type_map.items():
        fa = fa.withColumn(column, col(column).cast(new_type))

    # SSN processing
    pattern = r'^\d{3}-\d{2}-\d{4}$'
    fa = fa.withColumn(
        "SSN",
        when(col("SSN").rlike(pattern), col("SSN")).otherwise("___")
    )

    # Occupation processing
    fa = fa.withColumn(
        "Occupation",
        when(
            # after cleaning, if null or empty → set to "___"
            (col("Occupation").isNull()) | 
            (trim(regexp_replace(col("Occupation"), r"^_+|_+$", "")) == ""),
            "___"
        ).otherwise(
            # else keep the cleaned value
            trim(regexp_replace(col("Occupation"), r"^_+|_+$", ""))
        )
    )

    # Age processing
    # 1. Clean Age: strip spaces/underscores, cast to float
    fa = fa.withColumn(
        "Age",
        regexp_replace(trim(col("Age")), r"^_+|_+$", "").cast("float")
    )

    # 2. Compute average Age_ per Occupation_, filtering valid ages only
    avg_age_by_occ = (
        fa
        .filter((col("Age") >= 0) & (col("Age") <= 100))
        .groupBy("Occupation")
        .agg(avg("Age").alias("Age_avg"))
    )

    # 3. Join back to main dataframe
    fa = fa.join(avg_age_by_occ, on="Occupation", how="left")

    # 4. Replace outlier ages with group average
    fa = fa.withColumn(
        "Age",
        when((col("Age") >= 0) & (col("Age") <= 100), col("Age"))
        .otherwise(col("Age_avg"))
    )

    # 5. Round to integer
    fa = fa.withColumn("Age", spark_round(col("Age")).cast("int"))

    # 6. Drop helper column
    fa = fa.drop("Age_avg")

    # save silver table - IRL connect to database to write
    partition_name = "silver_FA_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    fa.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return fa


def process_silver_table_feature_clickstream(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    fc = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', fc.count())
    ###################
    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }
    column_type_map.update({f"fe_{i}": IntegerType() for i in range(1, 21)})

    for column, new_type in column_type_map.items():
        fc = fc.withColumn(column, col(column).cast(new_type))

    # get list of feature columns
    fe_cols = [c for c in fc.columns if c.startswith("fe")]

    # 1. Compute means for all feature columns at once
    mean_exprs = [spark_mean(col(c)).alias(c) for c in fe_cols ]
    mean_row = fc.select(*mean_exprs).collect()[0].asDict()

    # 2. Fill NaN/Null values with column means
    fc = fc.fillna(mean_row)

    # save silver table - IRL connect to database to write
    partition_name = "silver_FC_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    fc.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return fc

def process_silver_table_feature_financials(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    
    def ff_col_to_clean(df, col_name, dtype, clipmax=None):
        # Step 1: remove invalid chars (anything not digit or dot)
        df = df.withColumn(col_name, F.regexp_replace(col_name, "[^0-9.]", ""))

        # Step 2: cast to float — invalid will become null
        df = df.withColumn(col_name, F.col(col_name).cast(dtype))

        # Step 3: Set min annual income to be 0
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name) < 0, 0)
            .otherwise(F.col(col_name))
        )

        if clipmax is not None:
            # Step 4: Cap at clipmax
            df = df.withColumn(
                col_name,
                F.when(F.col(col_name) > clipmax, clipmax)
                .otherwise(F.col(col_name))
            )
        return df

    df = ff_col_to_clean(df, "Annual_Income", "float")
    df = ff_col_to_clean(df, "Monthly_Inhand_Salary", "float")
    df = ff_col_to_clean(df, "Num_Bank_Accounts", "int", clipmax=10)
    df = ff_col_to_clean(df, "Num_Credit_Card", "int", clipmax=10)
    df = ff_col_to_clean(df, "Num_Credit_Inquiries", "int", clipmax=200)
    df = ff_col_to_clean(df, "Interest_Rate", "float", clipmax=33)
    df = ff_col_to_clean(df, "Num_of_Loan", "int", clipmax=9)
    df = ff_col_to_clean(df, "Delay_from_due_date", "int", clipmax=57)
    df = ff_col_to_clean(df, "Num_of_Delayed_Payment", "int", clipmax=25)
    df = ff_col_to_clean(df, "Changed_Credit_Limit", "float")
    df = ff_col_to_clean(df, "Outstanding_Debt", "float")
    df = ff_col_to_clean(df, "Credit_Utilization_Ratio", "float")
    df = ff_col_to_clean(df, "Total_EMI_per_month", "float")
    df = ff_col_to_clean(df, "Amount_invested_monthly", "float")
    df = ff_col_to_clean(df, "Monthly_Balance", "float")

    ### convert credit history age
    # Step 1: Extract numeric parts using regex
    df = df.withColumn("Years",  F.regexp_extract(F.col("Credit_History_Age"), r'(\d+)\s+Years', 1))
    df = df.withColumn("Months", F.regexp_extract(F.col("Credit_History_Age"), r'(\d+)\s+Months', 1))
    # Step 2: Convert to integer (handle nulls / blanks)
    df = df.withColumn("Years",  F.when(F.col("Years") == "", 0).otherwise(F.col("Years").cast("int")))
    df = df.withColumn("Months", F.when(F.col("Months") == "", 0).otherwise(F.col("Months").cast("int")))
    # Step 3: Compute total months
    df = df.withColumn("Credit_History_Months", (F.col("Years") * 12 + F.col("Months")).cast("int"))
    # Step 4: Drop intermediate columns
    df = df.drop("Years", "Months")

    ### convert payment behaviour
    # Step 1 Define the column
    col = "Payment_Behaviour"

    # Step 2 Extract 'spend' and 'value' parts using regex
    df = (
        df.withColumn("Spend_Level",
            F.when(F.col(col).rlike("(?i)low_spent"), 1)
            .when(F.col(col).rlike("(?i)medium_spent"), 2)
            .when(F.col(col).rlike("(?i)high_spent"), 3)
            .otherwise(0)
        )
        .withColumn("Value_Level",
            F.when(F.col(col).rlike("(?i)small_value"), 1)
            .when(F.col(col).rlike("(?i)medium_value"), 2)
            .when(F.col(col).rlike("(?i)large_value"), 3)
            .otherwise(0)
        )
        .withColumn(
            "Credit_Mix",
            F.when((F.col("Credit_Mix").isin("_", "", "NA", "NaN")) | F.col("Credit_Mix").isNull(), "Unknown")
            .otherwise(F.initcap(F.col("Credit_Mix")))  # ensures "standard" → "Standard"
        )
        .withColumn(
            "Payment_of_Min_Amount",
            F.when(F.col("Payment_of_Min_Amount") == "Yes", "Yes")
            .when(F.col("Payment_of_Min_Amount") == "No", "No")
            .when(F.col("Payment_of_Min_Amount") == "NM", "Unknown")  # handle special case
            .otherwise("Unknown")
        )
    )

    ### loan types
    # all_loan_types extracted once from full feature financial data
    all_loan_types = ['Auto Loan',
    'Credit-Builder Loan',
    'Debt Consolidation Loan',
    'Home Equity Loan',
    'Mortgage Loan',
    'Payday Loan',
    'Personal Loan',
    'Student Loan']

    df = df.withColumn(
        "loan_list",
        F.split(F.regexp_replace(F.col("Type_of_Loan"), "and", ","), ",")
    )
    df = df.withColumn(
        "loan_list",
        F.expr("filter(transform(loan_list, x -> trim(x)), x -> x != '' and x != 'Not Specified')")
    )

    for loan in all_loan_types:
        df = df.withColumn(
            loan,
            F.when(F.array_contains(F.col("loan_list"), loan), F.lit(1)).otherwise(F.lit(0))
        )


    # sum across all loan columns
    df = df.withColumn("sum_loans", sum(F.col(c) for c in all_loan_types))

    # unseen = 1 if none of the known loan types are marked (sum == 0)
    df = df.withColumn("unseen", F.when(F.col("sum_loans") == 0, 1).otherwise(0))

    # optionally drop helper column
    df = df.drop("sum_loans")
    df = df.drop("loan_list")


    ### snapshot date
    # Step 1 Normalize common separators
    df = df.withColumn(
        "snapshot_date_str",
        F.regexp_replace(F.col("snapshot_date"), "[./]", "-")
    )

    # Step 2 Try multiple date formats
    df = df.withColumn(
        "snapshot_date",
        F.coalesce(
            F.to_date("snapshot_date_str", "yyyy-MM-dd"),
            F.to_date("snapshot_date_str", "dd-MM-yyyy"),
            F.to_date("snapshot_date_str", "MM-dd-yyyy"),
            F.to_date("snapshot_date_str", "yyyy/MM/dd")
        )
    )

    # save silver table - IRL connect to database to write
    partition_name = "silver_FF_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df
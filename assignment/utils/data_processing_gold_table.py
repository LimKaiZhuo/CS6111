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
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, NumericType
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_features_gold_table(snapshot_date_str, silver_parent_daily_directory, gold_feature_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    print('loaded from:', silver_parent_daily_directory,snapshot_date)

    fc = spark.read.parquet(f'{silver_parent_daily_directory}/feature_clickstream/silver_FC_{snapshot_date_str.replace('-','_')}.parquet')
    fa = spark.read.parquet(f'{silver_parent_daily_directory}/feature_attribute/silver_FA_{snapshot_date_str.replace('-','_')}.parquet')
    ff = spark.read.parquet(f'{silver_parent_daily_directory}/feature_financials/silver_FF_{snapshot_date_str.replace('-','_')}.parquet') 

    df = (fa
        .join(ff, on=["Customer_ID", "snapshot_date"], how="inner")
        .join(fc, on=["Customer_ID", "snapshot_date"], how="inner")
        )

    # all_loan_types extracted once from full feature financial data
    all_loan_types = ['Auto Loan',
    'Credit-Builder Loan',
    'Debt Consolidation Loan',
    'Home Equity Loan',
    'Mortgage Loan',
    'Payday Loan',
    'Personal Loan',
    'Student Loan',
    'unseen']

    # financial multipliers
    # FF
    df = df.withColumn("Debt_to_Income_Ratio", F.col("Outstanding_Debt") / F.col("Annual_Income"))
    df = df.withColumn("Delayed_Payment_Rate", F.col("Num_of_Delayed_Payment") / F.col("Num_of_Loan"))
    df = df.withColumn("EMI_to_Income_Ratio", F.col("Total_EMI_per_month") / F.col("Monthly_Inhand_Salary"))
    df = df.withColumn("Investment_Rate", F.col("Amount_invested_monthly") / F.col("Monthly_Inhand_Salary"))
    df = df.withColumn("Delayed_Payment_Rate", F.col("Num_of_Delayed_Payment") / F.col("Num_of_Loan"))
    df = df.withColumn("Loan_Diversity_Score", sum(F.col(c) for c in all_loan_types))

    # FC
    click_cols = [c for c in df.columns if c.startswith("fe_")]
    df = df.withColumn("Click_Total", sum(F.col(c) for c in click_cols))
    df = df.withColumn("Click_Mean", F.lit(1.0 / len(click_cols)) * sum(F.col(c) for c in click_cols))
    df = df.withColumn(
        "Click_Variance",
        sum((F.col(c) - F.col("Click_Mean"))**2 for c in click_cols) / F.lit(len(click_cols))
    )

    # cross features
    df = df.withColumn("Debt_to_Click_Ratio", F.col("Outstanding_Debt") / (F.col("Click_Total") + F.lit(1)))
    df = df.withColumn("Income_to_Utilization", F.col("Annual_Income") / (F.col("Credit_Utilization_Ratio") + F.lit(1)))
    df = df.withColumn("Investment_to_Clicks", F.col("Amount_invested_monthly") / (F.col("Click_Total") + F.lit(1)))
    df = df.withColumn("Age_to_Credit_Age_Ratio", F.col("Credit_History_Months") / (F.col("Age") * 12))
    df = df.withColumn("EMI_to_Debt_Ratio", F.col("Total_EMI_per_month") / (F.col("Outstanding_Debt") + F.lit(1)))
    df = df.withColumn("Debt_to_Salary_Interaction", F.col("Outstanding_Debt") / (F.col("Monthly_Inhand_Salary") + F.lit(1)))
    df = df.withColumn("Debt_Interest_Product", F.col("Outstanding_Debt") * F.col("Interest_Rate"))

    # non-linear transformations
    df = df.withColumn("Log_Debt", F.log1p(F.col("Outstanding_Debt")))
    df = df.withColumn("Log_Income", F.log1p(F.col("Annual_Income")))
    df = df.withColumn("Credit_Utilization_Sq", F.col("Credit_Utilization_Ratio")**2)
    df = df.withColumn("Credit_Utilization_Cube", F.col("Credit_Utilization_Ratio")**3)

    # threshold indicator variables (for linear regression)
    df = df.withColumn("High_Risk_Flag", F.when(F.col("Debt_to_Income_Ratio") > 0.8, 1).otherwise(0))
    df = df.withColumn("High_Interest_Flag", F.when(F.col("Interest_Rate") > 25, 1).otherwise(0))
    df = df.withColumn("Has_Multiple_Loans", F.when(F.col("Num_of_Loan") > 2, 1).otherwise(0))
    df = df.withColumn("Age_Bucket",
        F.when(F.col("Age") < 25, 0)
            .when(F.col("Age") < 40, 1)
            .when(F.col("Age") < 55, 2)
            .otherwise(3))
    df = df.withColumn("Debt_to_Income_Bin", F.when(F.col("Debt_to_Income_Ratio") < 0.3, 0)
                                                .when(F.col("Debt_to_Income_Ratio") < 0.7, 1)
                                                .when(F.col("Debt_to_Income_Ratio") < 1.0, 2)
                                                .otherwise(3))
    df = df.withColumn("Min_Payment_Flag",
                    F.when(F.col("Payment_of_Min_Amount") == "Yes", 1)
                        .when(F.col("Payment_of_Min_Amount") == "No", 0)
                        .otherwise(-1))
    df = df.withColumn(
        "PMET_Job_Indicator",
        F.when(F.col("Occupation").isin(["Engineer", "Manager", "Analyst", 'Lawyer', 'Scientist', 'Doctor', 'Accountant', 'Developer', 'Architect']), 1).otherwise(0)
    )
    df = df.withColumn("Is_Year_End", F.when(F.month("snapshot_date") == 12, 1).otherwise(0))

    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]

    for c in numeric_cols:
        df = df.withColumn(
            c,
            F.when(F.isnan(F.col(c)) | F.col(c).isNull(), 0).otherwise(F.col(c))
        )

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df


def process_features_gold_table__XGB_v1(snapshot_date_str, gold_feature_store_directory, gold_feature_M_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # select columns to save for linear regression V1 model
    sel_cols = ["Customer_ID", "snapshot_date", 
                'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 'Interest_Rate',
                'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
                'Monthly_Inhand_Salary', 'Age', 'Credit_History_Months',
                'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
                'Spend_Level', 'Value_Level',
                'Auto Loan',
                'Credit-Builder Loan',
                'Debt Consolidation Loan',
                'Home Equity Loan',
                'Mortgage Loan',
                'Payday Loan',
                'Personal Loan',
                'Student Loan',
                'unseen'] + [f'fe_{i}' for i in range(1,21)]
    
    sel_cols.extend(
        ['Debt_to_Income_Ratio', 'Delayed_Payment_Rate', 'EMI_to_Income_Ratio', 'Investment_Rate', 'Loan_Diversity_Score',
         'Click_Total', 'Click_Mean', 'Click_Variance',
         'Debt_to_Click_Ratio', 'Income_to_Utilization', 'Investment_to_Clicks', 'Age_to_Credit_Age_Ratio', 'EMI_to_Debt_Ratio', 'Debt_to_Salary_Interaction', 'Debt_Interest_Product',
         'Log_Debt', 'Log_Income', 'Credit_Utilization_Sq', 'Credit_Utilization_Cube',
         'High_Risk_Flag', 'High_Interest_Flag', 'Has_Multiple_Loans', 'Age_Bucket', 'Debt_to_Income_Bin', 'Min_Payment_Flag', 'PMET_Job_Indicator', 'Is_Year_End']
    )

    df = df.select(sel_cols)

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store__XGB_v1" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_M_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_features_gold_table__LR_v1(snapshot_date_str, gold_feature_store_directory, gold_feature_M_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # select columns to save for linear regression V1 model
    sel_cols = ["Customer_ID", "snapshot_date", 
                'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 'Interest_Rate',
                'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
                'Monthly_Inhand_Salary', 'Age', 'Credit_History_Months',
                'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
                'Spend_Level', 'Value_Level',
                'Auto Loan',
                'Credit-Builder Loan',
                'Debt Consolidation Loan',
                'Home Equity Loan',
                'Mortgage Loan',
                'Payday Loan',
                'Personal Loan',
                'Student Loan',
                'unseen']
    
    sel_cols.extend(
        ['Debt_to_Income_Ratio', 'Delayed_Payment_Rate', 'EMI_to_Income_Ratio', 'Investment_Rate', 'Loan_Diversity_Score',
         'Click_Total', 'Click_Mean', 'Click_Variance',
         'Debt_to_Click_Ratio', 'Income_to_Utilization', 'Investment_to_Clicks', 'Age_to_Credit_Age_Ratio', 'EMI_to_Debt_Ratio', 'Debt_to_Salary_Interaction', 'Debt_Interest_Product',
         'Log_Debt', 'Log_Income', 'Credit_Utilization_Sq', 'Credit_Utilization_Cube',
         'High_Risk_Flag', 'High_Interest_Flag', 'Has_Multiple_Loans', 'Age_Bucket', 'Debt_to_Income_Bin', 'Min_Payment_Flag', 'PMET_Job_Indicator', 'Is_Year_End']
    )

    # Drop due to multicollinearity or high correlation to engineered features
    drop_cols = [
    'Outstanding_Debt', 'Annual_Income', 'Monthly_Inhand_Salary',
    'Total_EMI_per_month', 'Amount_invested_monthly', 'Age'
    ]

    df = df.select(sel_cols)

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store__LR_v1" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_M_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df
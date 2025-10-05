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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table



# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# create bronze datalake
bronze_lms_directory = "datamart/bronze/lms/"

if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_lms_directory, spark)
    pass


# create bronze features datalake
bronze_dataset = {'data/features_attributes.csv': "datamart/bronze/features/features_attributes",
                   'data/feature_clickstream.csv': "datamart/bronze/features/feature_clickstream",
                     'data/features_financials.csv': "datamart/bronze/features/features_financials"}

for bronze_directory in bronze_dataset.values():
    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)

# run bronze features backfill
for data_dir, bronze_directory in bronze_dataset.items():
    for snapshot_date_str in dates_str_lst:
        utils.data_processing_bronze_table.process_bronze_table_features(data_dir, snapshot_date_str, bronze_directory, spark)
        pass


############################################################
# create silver datalake
silver_loan_daily_directory = "datamart/silver/loan_daily/"

if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

# run silver backfill
for date_str in dates_str_lst:
    bronze_lms_directory = "datamart/bronze/lms/"
    silver_loan_daily_directory = "datamart/silver/loan_daily/"
    utils.data_processing_silver_table.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
    bronze_lms_directory = "datamart/bronze/features/features_attributes/"
    silver_loan_daily_directory = "datamart/silver/feature_attribute/"
    utils.data_processing_silver_table.process_silver_table_feature_attribute(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
    bronze_lms_directory = "datamart/bronze/features/feature_clickstream/"
    silver_loan_daily_directory = "datamart/silver/feature_clickstream/"
    utils.data_processing_silver_table.process_silver_table_feature_clickstream(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
    bronze_lms_directory = "datamart/bronze/features/features_financials/"
    silver_loan_daily_directory = "datamart/silver/feature_financials/"
    utils.data_processing_silver_table.process_silver_table_feature_financials(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)

# create bronze datalake
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    silver_loan_daily_directory = "datamart/silver/loan_daily/"
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)
    silver_loan_daily_directory = "datamart/silver/"
    utils.data_processing_gold_table.process_features_gold_table(date_str, "datamart/silver/", "datamart/gold/feature_store/", spark)
    utils.data_processing_gold_table.process_features_gold_table__LR_v1(date_str, "datamart/gold/feature_store/", "datamart/gold/feature_store__LR_v1/", spark)
    utils.data_processing_gold_table.process_features_gold_table__XGB_v1(date_str, "datamart/gold/feature_store/", "datamart/gold/feature_store__XGB_v1/", spark)

folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()



    
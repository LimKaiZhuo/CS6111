import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
from pathlib import Path
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    current_date = datetime(start_date.year, start_date.month, 1)
    dates = []
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    return dates


def run_pipeline(start_date_str: str, end_date_str: str):
    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
    print(dates_str_lst)

    bronze_lms_directory = "datamart/bronze/lms/"
    os.makedirs(bronze_lms_directory, exist_ok=True)
    for date_str in dates_str_lst:
        utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_lms_directory, spark)

    bronze_dataset = {
        "data/features_attributes.csv": ("datamart/bronze/features/features_attributes", True),
        "data/feature_clickstream.csv": ("datamart/bronze/features/feature_clickstream", False),
        "data/features_financials.csv": ("datamart/bronze/features/features_financials", True),
    }
    for bronze_directory, _ in bronze_dataset.values():
        os.makedirs(bronze_directory, exist_ok=True)

    for data_dir, (bronze_directory, overwrite_table) in bronze_dataset.items():
        for snapshot_date_str in dates_str_lst:
            utils.data_processing_bronze_table.process_bronze_table_features(
                data_dir, snapshot_date_str, bronze_directory, spark, overwrite_table=overwrite_table
            )

    for date_str in dates_str_lst:
        utils.data_processing_silver_table.process_silver_table(
            date_str, "datamart/bronze/lms/", "datamart/silver/loan_daily/", spark
        )
        utils.data_processing_silver_table.process_silver_table_feature_attribute(
            date_str, "datamart/bronze/features/features_attributes/", "datamart/silver/feature_attribute/", spark
        )
        utils.data_processing_silver_table.process_silver_table_feature_clickstream(
            date_str, "datamart/bronze/features/feature_clickstream/", "datamart/silver/feature_clickstream/", spark
        )
        utils.data_processing_silver_table.process_silver_table_feature_financials(
            date_str, "datamart/bronze/features/features_financials/", "datamart/silver/feature_financials/", spark
        )

    gold_label_store_directory = "datamart/gold/label_store/"
    os.makedirs(gold_label_store_directory, exist_ok=True)

    for date_str in dates_str_lst:
        utils.data_processing_gold_table.process_labels_gold_table(
            date_str, "datamart/silver/loan_daily/", gold_label_store_directory, spark, dpd=30, mob=6
        )
        utils.data_processing_gold_table.process_features_gold_table(
            date_str, "datamart/silver/", "datamart/gold/feature_store/", spark
        )
        utils.data_processing_gold_table.process_features_gold_table__LR_v1(
            date_str, "datamart/gold/feature_store/", "datamart/gold/feature_store__LR_v1/", spark
        )
        utils.data_processing_gold_table.process_features_gold_table__XGB_v1(
            date_str, "datamart/gold/feature_store/", "datamart/gold/feature_store__XGB_v1/", spark
        )

    files_list = [
        gold_label_store_directory + os.path.basename(f)
        for f in glob.glob(os.path.join(gold_label_store_directory, "*"))
    ]
    df = spark.read.option("header", "true").parquet(*files_list)
    print("row_count:", df.count())
    df.show()

    spark.stop()


def main():
    parser = argparse.ArgumentParser(description="Backfill bronze/silver/gold pipelines over a date range.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD start date (inclusive).")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD end date (inclusive).")
    args = parser.parse_args()

    run_pipeline(args.start_date, args.end_date)


if __name__ == "__main__":
    main()

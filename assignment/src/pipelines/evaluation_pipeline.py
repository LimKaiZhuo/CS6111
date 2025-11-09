#!/usr/bin/env python3
"""
Inference evaluation pipeline: rebuilds labels for a snapshot and
compares them with saved inference outputs using AUC.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.data_processing_bronze_table import process_bronze_table
from utils.data_processing_silver_table import process_silver_table
from utils.data_processing_gold_table import process_labels_gold_table


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_with_trailing_slash(path: Path) -> str:
    posix = path.as_posix()
    return posix if posix.endswith("/") else f"{posix}/"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate inference output against freshly built labels for a snapshot."
    )
    parser.add_argument("--snapshot-date", required=True, help="Snapshot date in YYYY-MM-DD format.")
    parser.add_argument("--inference-csv", required=True, help="Path to inference output CSV.")
    parser.add_argument("--raw-data-root", default="data", help="Root directory for raw CSV inputs.")
    parser.add_argument("--datamart-root", default="datamart", help="Root directory for datamart outputs.")
    parser.add_argument("--dpd", type=int, default=30, help="Days past due threshold for label creation.")
    parser.add_argument("--mob", type=int, default=6, help="Months-on-book filter for label creation.")
    return parser


def _load_inference_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"Customer_ID", "default_probability", "prediction"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Inference CSV missing required columns: {missing}")
    return df


def _build_labels(
    spark: SparkSession,
    snapshot_date: str,
    datamart_root: Path,
    dpd: int,
    mob: int,
) -> pd.DataFrame:
    bronze_lms_dir = _format_with_trailing_slash(_ensure_dir(datamart_root / "bronze" / "lms"))
    process_bronze_table(snapshot_date, bronze_lms_dir, spark)

    silver_loan_daily_dir = _format_with_trailing_slash(_ensure_dir(datamart_root / "silver" / "loan_daily"))
    process_silver_table(
        snapshot_date,
        bronze_lms_dir,
        silver_loan_daily_dir,
        spark,
    )

    gold_label_dir = _format_with_trailing_slash(_ensure_dir(datamart_root / "gold" / "label_store"))
    process_labels_gold_table(
        snapshot_date,
        silver_loan_daily_dir,
        gold_label_dir,
        spark,
        dpd=dpd,
        mob=mob,
    )

    snapshot_token = snapshot_date.replace("-", "_")
    label_path = Path(gold_label_dir) / f"gold_label_store_{snapshot_token}.parquet"
    labels_df = spark.read.parquet(str(label_path)).select("Customer_ID", "label")
    return labels_df.toPandas()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    inference_path = Path(args.inference_csv).resolve()
    if not inference_path.exists():
        raise FileNotFoundError(f"Inference CSV not found at {inference_path}")

    datamart_root = Path(args.datamart_root).resolve()

    spark: SparkSession = SparkSession.builder.appName("evaluation_pipeline").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        inference_df = _load_inference_csv(inference_path)
        labels_df = _build_labels(
            spark,
            args.snapshot_date,
            datamart_root,
            args.dpd,
            args.mob,
        )
        merged = inference_df.merge(labels_df, on="Customer_ID", how="inner")
        if merged.empty:
            raise ValueError("No overlapping Customer_ID between inference output and labels.")

        missing_labels = len(labels_df) - len(merged)
        if missing_labels:
            logging.info("Labels unmatched with inference output: %d", missing_labels)
        
        # Keep all original columns from inference_df and add the new 'label'
        all_cols = list(inference_df.columns) + ["label"]
        merged_with_labels = merged.loc[:, all_cols]
        labels_output_path = inference_path.with_name(f"{inference_path.stem}_with_labels{inference_path.suffix}")
        merged_with_labels.to_csv(labels_output_path, index=False)

        auc_score = roc_auc_score(merged["label"], merged["default_probability"])

        auc_output_path = inference_path.with_name(f"{inference_path.stem}_AUC{inference_path.suffix}")
        auc_df = pd.DataFrame({"metric": ["AUC"], "value": [auc_score], "records": [len(merged)]})
        auc_df.to_csv(auc_output_path, index=False)

        metadata = {
            "snapshot_date": args.snapshot_date,
            "inference_csv": str(inference_path),
            "labels_csv": str(labels_output_path),
            "label_params": {"dpd": args.dpd, "mob": args.mob},
            "records": int(len(merged)),
            "auc": float(auc_score),
            "evaluated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        metadata_path = inference_path.with_name(f"{inference_path.stem}_AUC.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logging.info(
            "Evaluation completed. AUC = %.4f on %d records. Saved results to %s, %s, and %s",
            auc_score,
            len(merged),
            auc_output_path,
            metadata_path,
            labels_output_path,
        )
        print(
            f"AUC: {auc_score:.4f} "
            f"(saved inference+labels to {labels_output_path}, metrics to {auc_output_path} and {metadata_path})"
        )
        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
"""
python src/pipelines/evaluation_pipeline.py \
  --snapshot-date 2024-10-01 \
  --inference-csv reports/inference_20241001.csv \
  --datamart-root datamart

"""

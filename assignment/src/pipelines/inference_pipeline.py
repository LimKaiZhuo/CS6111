#!/usr/bin/env python3
"""
End-to-end inference runner for the CS6111 XGBoost model.

Given a monthly snapshot (YYYY-MM-DD), the script:
1. Rebuilds the bronze, silver, and gold feature pipelines for that period.
2. Loads the pre-trained best model from the training results metadata.
3. Scores every customer in the gold feature table and returns probabilities
   plus hard predictions.

Example
-------
python src/pipelines/inference_pipeline.py \
  --snapshot-date 2024-10-01 \
  --results-json outputs/results.json \
  --output-csv reports/inference_20241001.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import pyspark.sql
from pyspark.sql import SparkSession
import xgboost as xgb

# ---------------------------------------------------------------------------
# Project imports (ensure repo root is on PYTHONPATH)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.data_processing_bronze_table import (
    process_bronze_table,
    process_bronze_table_features,
)
from utils.data_processing_silver_table import (
    process_silver_table,
    process_silver_table_feature_attribute,
    process_silver_table_feature_clickstream,
    process_silver_table_feature_financials,
)
from utils.data_processing_gold_table import (
    process_features_gold_table,
    process_features_gold_table__XGB_v1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    """Create the directory if needed and return the same Path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_with_trailing_slash(path: Path) -> str:
    """Return POSIX string with a single trailing slash."""
    as_posix = path.as_posix()
    return as_posix if as_posix.endswith("/") else f"{as_posix}/"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bronze, silver, and gold pipelines before scoring the XGBoost model."
    )
    parser.add_argument(
        "--snapshot-date",
        required=True,
        help="Snapshot date in YYYY-MM-DD format (first of month recommended).",
    )
    parser.add_argument(
        "--results-json",
        default="outputs/results.json",
        help="Path to the training results metadata JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--model-uri",
        help="Optional override for the model URI. Defaults to the stage 2 URI stored in the results JSON.",
    )
    parser.add_argument(
        "--raw-data-root",
        default="data",
        help="Directory holding the raw CSV inputs (default: %(default)s).",
    )
    parser.add_argument(
        "--datamart-root",
        default="datamart",
        help="Root directory for bronze/silver/gold outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for converting scores into hard predictions.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path to save the scored dataset as CSV.",
    )
    parser.add_argument(
        "--skip-backfill",
        default=False,
        action="store_true",
        help="Skip backfilling the bronze/silver/gold tables."
    )
    return parser


def _determine_output_path(snapshot_date: str, requested_path: Optional[str]) -> Path:
    """Return the output path, defaulting to reports/inference_<YYYYMMDD>.csv."""
    if requested_path:
        path = Path(requested_path)
    else:
        safe_snapshot = snapshot_date.replace("-", "")
        path = PROJECT_ROOT / "reports" / f"inference_{safe_snapshot}.csv"
    return path.resolve()


def _load_label_membership(spark: SparkSession, label_store_root: Path) -> pd.DataFrame:
    """
    Build a lookup mapping each Customer_ID to the most recent snapshot month where it appears.

    The function scans all parquet datasets under the label store directory, records the snapshot
    month derived from the file name, and returns a DataFrame with columns:
        - Customer_ID
        - label_snapshot_month (string YYYY_MM_DD derived from the dataset name)
    """
    if not label_store_root.exists():
        logging.warning("Label store directory %s does not exist; skipping label join.", label_store_root)
        return pd.DataFrame(columns=["Customer_ID", "label_snapshot_month"])

    label_frames: list[pd.DataFrame] = []
    for dataset in sorted(label_store_root.glob("gold_label_store_*.parquet")):
        snapshot_month = dataset.stem.replace("gold_label_store_", "").replace("_", "-")
        sdf = spark.read.parquet(str(dataset)).select("Customer_ID").dropna()
        pdf = sdf.toPandas()
        if pdf.empty:
            continue
        pdf = pdf.drop_duplicates(subset=["Customer_ID"])
        pdf["label_snapshot_month"] = snapshot_month
        label_frames.append(pdf[["Customer_ID", "label_snapshot_month"]])

    if not label_frames:
        logging.info("No label membership data found under %s.", label_store_root)
        return pd.DataFrame(columns=["Customer_ID", "label_snapshot_month"])

    merged = pd.concat(label_frames, ignore_index=True)
    merged.sort_values("label_snapshot_month", inplace=True)
    merged = merged.drop_duplicates(subset=["Customer_ID"], keep="last")
    return merged


def _run_bronze(spark: SparkSession, snapshot_date: str, raw_root: Path, datamart_root: Path) -> None:
    logging.info("Running bronze pipelines for %s", snapshot_date)

    bronze_lms_dir = _format_with_trailing_slash(_ensure_dir(datamart_root / "bronze" / "lms"))

    # Ingest loan daily snapshot
    """
    Do not run bronze label table
    process_bronze_table(snapshot_date, bronze_lms_dir, spark)
    """
    # Feature datasets
    bronze_features_root = datamart_root / "bronze" / "features"
    dataset_map = {
        raw_root / "features_attributes.csv": (
            _format_with_trailing_slash(_ensure_dir(bronze_features_root / "features_attributes")),
            True,
        ),
        raw_root / "feature_clickstream.csv": (
            _format_with_trailing_slash(_ensure_dir(bronze_features_root / "feature_clickstream")),
            False,
        ),
        raw_root / "features_financials.csv": (
            _format_with_trailing_slash(_ensure_dir(bronze_features_root / "features_financials")),
            True,
        ),
    }

    for csv_path, (target_dir, overwrite) in dataset_map.items():
        logging.info("Processing bronze feature file %s (overwrite=%s)", csv_path, overwrite)
        process_bronze_table_features(
            str(csv_path),
            snapshot_date,
            target_dir,
            spark,
            overwrite_table=overwrite,
        )


def _run_silver(spark: SparkSession, snapshot_date: str, datamart_root: Path) -> None:
    logging.info("Running silver pipelines for %s", snapshot_date)

    bronze_root = datamart_root / "bronze"
    silver_root = datamart_root / "silver"

    """
    Do not run label's silver table
    silver_loan_daily_dir = _format_with_trailing_slash(_ensure_dir(silver_root / "loan_daily"))
    process_silver_table(
        snapshot_date,
        _format_with_trailing_slash(bronze_root / "lms"),
        silver_loan_daily_dir,
        spark,
    )
    """

    process_silver_table_feature_attribute(
        snapshot_date,
        _format_with_trailing_slash(bronze_root / "features" / "features_attributes"),
        _format_with_trailing_slash(_ensure_dir(silver_root / "feature_attribute")),
        spark,
    )
    process_silver_table_feature_clickstream(
        snapshot_date,
        _format_with_trailing_slash(bronze_root / "features" / "feature_clickstream"),
        _format_with_trailing_slash(_ensure_dir(silver_root / "feature_clickstream")),
        spark,
    )
    process_silver_table_feature_financials(
        snapshot_date,
        _format_with_trailing_slash(bronze_root / "features" / "features_financials"),
        _format_with_trailing_slash(_ensure_dir(silver_root / "feature_financials")),
        spark,
    )


def _run_gold_features(spark: SparkSession, snapshot_date: str, datamart_root: Path) -> Path:
    logging.info("Running gold feature pipeline for %s", snapshot_date)

    silver_root = _ensure_dir(datamart_root / "silver")
    gold_feature_root = _format_with_trailing_slash(_ensure_dir(datamart_root / "gold" / "feature_store"))
    gold_feature_xgb_root = _format_with_trailing_slash(_ensure_dir(datamart_root / "gold" / "feature_store__XGB_v1"))

    process_features_gold_table(
        snapshot_date,
        silver_root.as_posix(),
        gold_feature_root,
        spark,
    )
    process_features_gold_table__XGB_v1(
        snapshot_date,
        gold_feature_root,
        gold_feature_xgb_root,
        spark,
    )

    snapshot_token = snapshot_date.replace("-", "_")
    feature_path = Path(gold_feature_xgb_root) / f"gold_feature_store__XGB_v1{snapshot_token}.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"Expected gold feature parquet at {feature_path}")
    return feature_path


def _load_features(feature_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Loading gold features from %s", feature_path)
    spark = pyspark.sql.SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("Spark session is not active while loading features.")

    features_df = spark.read.parquet(str(feature_path))
    pdf = features_df.toPandas()

    id_cols = [c for c in ["Customer_ID", "snapshot_date"] if c in pdf.columns]
    feature_cols = [c for c in pdf.columns if c not in id_cols]

    if not feature_cols:
        raise ValueError("No usable feature columns found in gold feature table.")

    X = pdf[feature_cols].copy()
    identifiers = pdf[id_cols].copy() if id_cols else pd.DataFrame(index=pdf.index)
    return identifiers, X


def _load_model(model_uri: str) -> xgb.Booster | xgb.XGBModel:
    logging.info("Loading model from %s", model_uri)
    return mlflow.xgboost.load_model(model_uri)


def _predict(model: xgb.Booster | xgb.XGBModel, X: pd.DataFrame, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(X, enable_categorical=True)
        proba = model.predict(dmatrix)
    else:
        proba = model.predict_proba(X)[:, 1]
    proba = np.asarray(proba, dtype=float)
    preds = (proba >= threshold).astype(int)
    return proba, preds


def _load_results_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Results metadata JSON not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    snapshot_date = args.snapshot_date
    raw_root = Path(args.raw_data_root).resolve()
    datamart_root = Path(args.datamart_root).resolve()
    results_path = Path(args.results_json).resolve()

    results_meta = _load_results_metadata(results_path)
    stage2_uri = results_meta.get("stage2_model_artifact_uri")
    if not stage2_uri:
        raise ValueError(f"'stage2_model_artifact_uri' missing in results JSON: {results_path}")

    selected_features = results_meta.get("selected_features")
    if not selected_features:
        raise ValueError(f"'selected_features' missing in results JSON: {results_path}")
    logging.info(f"Features has length {len(selected_features)}. {selected_features}")
    logging.info("Starting Spark session for inference.")
    spark: SparkSession = SparkSession.builder.appName("xgb_inference").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        _run_bronze(spark, snapshot_date, raw_root, datamart_root)
        _run_silver(spark, snapshot_date, datamart_root)
        feature_path = _run_gold_features(spark, snapshot_date, datamart_root)

        identifiers, X = _load_features(feature_path)
        logging.info("Loaded X dataframe with shape: %s", X.shape)

        missing = [col for col in selected_features if col not in X.columns]
        if missing:
            raise KeyError(f"Selected features missing from gold table: {missing}")
        X_subset = X.loc[:, selected_features].copy()

        model_uri = args.model_uri or stage2_uri
        logging.info("Using model URI: %s", model_uri)
        model = _load_model(model_uri)
        logging.info(f'Expected model features: {model.get_booster().feature_names}')

        proba, preds = _predict(model, X_subset, args.threshold)

        # Combine identifiers, features, and predictions
        output = pd.concat([identifiers.reset_index(drop=True),
                            X_subset.reset_index(drop=True)], axis=1)
        output["default_probability"] = proba
        output["prediction"] = preds

        labels_lookup = _load_label_membership(spark, datamart_root / "gold" / "label_store")
        if not labels_lookup.empty:
            output = output.merge(labels_lookup, on="Customer_ID", how="left")
        else:
            output["label_snapshot_month"] = pd.NA

        logging.info(
            "Scored %d rows (mean probability %.4f).",
            len(output),
            float(proba.mean()) if len(proba) else float("nan"),
        )

        out_path = _determine_output_path(snapshot_date, args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(out_path, index=False)
        logging.info("Inference output written to %s", out_path)

        if not output.empty:
            print(output.head().to_markdown(index=False))
        else:
            print("No records scored.")

        return 0

    finally:
        spark.stop()
        logging.info("Spark session closed.")


if __name__ == "__main__":
    raise SystemExit(main())
"""
python src/pipelines/inference_pipeline.py \
  --snapshot-date 2024-10-01 \
  --results-json outputs/results_20251029_165223.json \
  --raw-data-root data \
  --datamart-root datamart \
  --threshold 0.5 \
  --output-csv reports/inference_20241001.csv

"""

#!/usr/bin/env python3
"""
Generate Evidently monitoring reports for data and model performance drift.

This script compares a reference dataset (from training) with a current dataset
(from inference) to detect drift. It generates:
1. An HTML report visualizing the drift analysis.
2. A JSON file summarizing the test suite results (e.g., pass/fail for drift).

Example
-------
python src/pipelines/generate_monitoring_report.py \
  --reference-csv /path/to/reference_with_predictions.csv \
  --current-csv /path/to/current_with_predictions_and_labels.csv \
  --report-html /path/to/report.html \
  --summary-json /path/to/summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sklearn.metrics import roc_auc_score
import pandas as pd
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.metric_preset import ClassificationPreset
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnValuePlot, Comment
from evidently.test_preset import DataDriftTestPreset as DataDriftTestPreset_
from evidently.test_preset import BinaryClassificationTestPreset as BinaryClassificationTestPreset_
from evidently.pipeline.column_mapping import ColumnMapping



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Evidently data and model drift reports."
    )
    parser.add_argument(
        "--reference-csv",
        required=True,
        help="Path to the reference data (CSV or Parquet).",
    )
    parser.add_argument(
        "--current-csv",
        required=True,
        help="Path to the current data (CSV or Parquet).",
    )
    parser.add_argument(
        "--report-html",
        required=True,
        help="Path to save the output HTML report.",
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Path to save the JSON summary of test results.",
    )
    parser.add_argument(
        "--results-json",
        required=True,
        help="Path to the training results metadata JSON with selected features.",
    )
    parser.add_argument(
        "--target-col",
        default="label",
        help="Name of the target column (ground truth).",
    )
    parser.add_argument(
        "--prediction-col",
        default="default_probability",
        help="Name of the prediction/score column.",
    )
    parser.add_argument(
        "--timestamp-col",
        default="feature_snapshot_date",
        help="Name of the timestamp column for time-series analysis.",
    )
    return parser

def _read_data(file_path_str: str) -> pd.DataFrame:
    """Reads data from CSV or Parquet file based on extension."""
    file_path = Path(file_path_str)
    if file_path.suffix.lower() == ".parquet":
        return pd.read_parquet(file_path)
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported file format: {file_path.suffix}. Please use .csv or .parquet.")


def main() -> int:
    """Main entrypoint for the monitoring report generation script."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("evidently_monitoring")

    logger.info("Loading reference data from %s", args.reference_csv)    
    reference_df = _read_data(args.reference_csv)

    logger.info("Loading current data from %s", args.current_csv)
    current_df = _read_data(args.current_csv)
    
    logger.info("Loading selected features from %s", args.results_json)
    with Path(args.results_json).open("r", encoding="utf-8") as f:
        results_meta = json.load(f)
    selected_features = results_meta.get("selected_features")
    if not selected_features:
        raise ValueError(f"Could not find 'selected_features' in {args.results_json}")

    # Ensure required columns exist
    required_cols = {args.target_col, args.prediction_col, args.timestamp_col}
    if not required_cols.issubset(reference_df.columns):
        raise ValueError(f"Reference data missing columns: {required_cols - set(reference_df.columns)}")
    if not {args.target_col, args.prediction_col}.issubset(current_df.columns):
        raise ValueError(f"Current data missing columns: {required_cols - set(current_df.columns)}")

    # --- Determine the final set of features present in BOTH dataframes ---
    common_features = list(set(selected_features) & set(reference_df.columns) & set(current_df.columns))
    if not common_features:
        raise ValueError("No common features found between reference and current dataframes.")
    logger.info(f"Found {len(common_features)} common features for monitoring.")

    # --- Subset dataframes to only the common features + target/prediction ---
    final_cols_to_keep = common_features + [args.target_col, args.prediction_col]
    if args.timestamp_col in reference_df.columns:
        final_cols_to_keep.append(args.timestamp_col)

    reference_df = reference_df[final_cols_to_keep]
    # current_df does not have the timestamp column, so we exclude it from its `final_cols_to_keep`
    current_df = current_df[common_features + [args.target_col, args.prediction_col]]

    # --- Manually create data for the AUC time trend plot ---
    logger.info("Calculating AUC for each month in reference data.")
    reference_df[args.timestamp_col] = pd.to_datetime(reference_df[args.timestamp_col])
    
    auc_by_month = (
        reference_df.groupby(args.timestamp_col)
        .apply(lambda x: roc_auc_score(x[args.target_col], x[args.prediction_col]))
        .reset_index(name="auc")
    )
    auc_by_month = auc_by_month.sort_values(args.timestamp_col)

    logger.info("Calculating AUC for current data.")
    current_auc = roc_auc_score(current_df[args.target_col], current_df[args.prediction_col])
    
    # Create a new row for the current data's AUC, placing it one month after the last reference month
    latest_month = auc_by_month[args.timestamp_col].max()
    current_month_timestamp = latest_month + pd.DateOffset(months=1)
    current_auc_row = pd.DataFrame([{args.timestamp_col: current_month_timestamp, "auc": current_auc}])
    
    # Combine historical and current AUCs into a single dataframe for plotting
    auc_trend_df = pd.concat([auc_by_month, current_auc_row], ignore_index=True)
    # Convert timestamp to string for clean plotting labels
    auc_trend_df[args.timestamp_col] = auc_trend_df[args.timestamp_col].dt.strftime('%Y-%m')

    # --- Manually create data for the Label Percentage time trend plot ---
    logger.info("Calculating percentage of positive labels over time.")
    # Calculate for reference data
    ref_label_perc = (
        reference_df.groupby(args.timestamp_col)[args.target_col]
        .apply(lambda x: (x == 1).mean() * 100)
        .reset_index(name="reference")
    )
    ref_label_perc = ref_label_perc.sort_values(args.timestamp_col)

    # Calculate for current data
    current_label_perc_val = (current_df[args.target_col] == 1).mean() * 100
    current_label_row = pd.DataFrame([{
        args.timestamp_col: current_month_timestamp,
        "current": current_label_perc_val
    }])

    # Combine into a single dataframe for plotting, merging on timestamp
    label_trend_df = pd.merge(ref_label_perc, current_label_row, on=args.timestamp_col, how="outer")
    label_trend_df = label_trend_df.sort_values(args.timestamp_col)
    # Convert timestamp to string for clean plotting labels
    label_trend_df[args.timestamp_col] = label_trend_df[args.timestamp_col].dt.strftime('%Y-%m')


    column_mapping = ColumnMapping(
        target=args.target_col,
        prediction=args.prediction_col,
        numerical_features=[col for col in common_features
                            if pd.api.types.is_numeric_dtype(reference_df[col])
                            and col != args.timestamp_col],
        categorical_features=[col for col in common_features
                              if not pd.api.types.is_numeric_dtype(reference_df[col])
                              and col != args.timestamp_col],
    )

    logger.info("Defining drift report and test suite")
    # A Report generates plots and visualizations.
    # We use ClassificationPreset for a full comparison of model quality.
    drift_report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset(),
    ])

    # --- Generate the custom plot in a separate, temporary report ---
    label_trend_report = Report(metrics=[
        Comment("Positive Label Percentage Over Time"),
        ColumnValuePlot(column_name="reference"),
        ColumnValuePlot(column_name="current"),
    ])
    label_trend_report.run(
        reference_data=label_trend_df,
        current_data=label_trend_df,
        column_mapping=ColumnMapping(datetime_features=[args.timestamp_col]),
    )
    # --- Add the generated widgets from the temp report to the main report ---
    for widget in label_trend_report.as_dict()["widgets"]:
        drift_report.add_widget(widget)

    # --- Define and run the main TestSuite ---
    drift_test_suite = TestSuite(tests=[
        DataDriftTestPreset_(),
        BinaryClassificationTestPreset_(),
    ])

    logger.info("Running drift analysis")
    # Run the main analysis on the actual data. The custom plot is already added.
    drift_report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    # The test suite does not need the combined data, it compares reference vs current
    drift_test_suite.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    report_path = Path(args.report_html)
    summary_path = Path(args.summary_json)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving HTML report to %s", report_path)
    drift_report.save_html(str(report_path))

    logger.info("Saving JSON summary to %s", summary_path)
    summary = drift_test_suite.as_dict()
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not summary["summary"]["all_passed"]:
        logger.warning("Drift detected! Some tests failed.")
    else:
        logger.info("No drift detected. All tests passed.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
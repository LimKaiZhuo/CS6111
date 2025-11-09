#!/usr/bin/env python3
"""
Generate a custom model monitoring report using Plotly.

This script compares a reference dataset (from training) with a current dataset
(from inference) to detect drift and generates a single, self-contained HTML file.

Analyses included:
1. Feature distribution drift plots (histograms for numerical, bar charts for categorical).
2. Overall model performance (AUC) comparison.
3. Time trend of AUC performance by month.

Example
-------
python src/pipelines/generate_plotly_report.py \
  --reference-csv /path/to/reference.parquet \
  --current-csv /path/to/current.csv \
  --results-json /path/to/results.json \
  --report-html /path/to/plotly_report.html
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import roc_auc_score
from dateutil.relativedelta import relativedelta

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a custom model monitoring report with Plotly."
    )
    parser.add_argument("--reference-csv", required=True, help="Path to the reference data.")
    parser.add_argument("--current-csv", required=True, help="Path to the current data.")
    parser.add_argument("--report-html", required=True, help="Path to save the output HTML report.")
    parser.add_argument("--plots-dir", required=True, help="Path to save individual plot HTML files for alerts.")
    parser.add_argument("--summary-json", required=True, help="Path to save the JSON summary of drift and AUC results.")
    parser.add_argument("--results-json", required=True, help="Path to the training results metadata JSON.")
    parser.add_argument("--target-col", default="label", help="Name of the target column.")
    parser.add_argument("--prediction-col", default="default_probability", help="Name of the prediction/score column.")
    parser.add_argument("--timestamp-col", default="feature_snapshot_date", help="Name of the timestamp column.")
    return parser

def _read_data(file_path_str: str) -> pd.DataFrame:
    """Reads data from CSV or Parquet file based on extension."""
    file_path = Path(file_path_str)
    if file_path.suffix.lower() == ".parquet":
        return pd.read_parquet(file_path)
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported file format: {file_path.suffix}. Please use .csv or .parquet.")

def create_feature_drift_plots(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, features: list[str], plots_dir: str
) -> tuple[list[go.Figure], list[dict]]:
    """Generates distribution plots and drift test results for each feature."""
    figs = []
    drift_results = []
    for feature in features:
        if pd.api.types.is_numeric_dtype(reference_df[feature]):
            # --- Numerical Feature Drift Detection (K-S Test) ---
            ks_stat, p_value = ks_2samp(reference_df[feature].dropna(), current_df[feature].dropna())
            drift_detected = bool(p_value < 0.05)
            title = f"Distribution for: {feature}<br>Drift Detected: {drift_detected} (p-value: {p_value:.4f})"
            if drift_detected:
                title = f"<b style='color:red;'>{title}</b>"
            drift_results.append({
                "feature": feature,
                "test": "Kolmogorov-Smirnov",
                "p_value": float(p_value),
                "drift_detected": bool(drift_detected),
            })

            # To avoid color mixing from overlapping transparent areas, we will plot
            # the outline of the histograms (a step plot) instead of filled bars.
            # First, determine common bins for both distributions.
            combined_data = pd.concat([reference_df[feature].dropna(), current_df[feature].dropna()])
            # Use a try-except block to handle cases where 'auto' binning fails on skewed data.
            try:
                auto_bins = np.histogram_bin_edges(combined_data, bins='auto')
                bins = 'auto' if len(auto_bins) <= 251 else 250
            except ValueError:
                bins = 250

            # Calculate histogram values. np.histogram returns the bin edges, which we need for plotting.
            ref_hist, ref_bins = np.histogram(reference_df[feature].dropna(), bins=bins, density=True)
            cur_hist, cur_bins = np.histogram(current_df[feature].dropna(), bins=bins, density=True)

            # Create the step plot traces.
            fig = go.Figure()
            
            # Current: Red filled area with transparency and no outline
            fig.add_trace(go.Scatter(
                x=cur_bins, y=cur_hist,
                fill='tozeroy',
                mode='lines',
                line_shape='hv',
                line=dict(width=0), # Hides the line
                fillcolor='rgba(255, 0, 0, 0.4)', # Red fill with transparency
                name='Current'
            ))

            # Reference: Smooth blue line, no fill
            bin_centers = (ref_bins[:-1] + ref_bins[1:]) / 2
            fig.add_trace(go.Scatter(x=bin_centers, y=ref_hist, mode='lines', line_shape='spline', name='Reference', line=dict(color='blue')))

            fig.update_layout(
                title_text=title,
                xaxis_title_text=feature,
                yaxis_title_text="Density",
            )
        else:
            # --- Categorical Feature Drift Detection (Chi-squared Test) ---
            # Create a contingency table of value counts
            contingency_table = pd.DataFrame({
                'reference': reference_df[feature].value_counts(),
                'current': current_df[feature].value_counts(),
            }).fillna(0)

            # Perform the test
            try:
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                drift_detected = bool(p_value < 0.05)
                title = f"Distribution for: {feature}<br>Drift Detected: {drift_detected} (p-value: {p_value:.4f})"
                if drift_detected:
                    title = f"<b style='color:red;'>{title}</b>"
                drift_results.append({
                    "feature": feature,
                    "test": "Chi-squared",
                    "p_value": float(p_value),
                    "drift_detected": bool(drift_detected),
                })
            except ValueError:
                # Test cannot be performed (e.g., all values are the same)
                title = f"Distribution for: {feature}<br>Drift Test Not Applicable"
                drift_results.append({"feature": feature, "test": "Chi-squared", "p_value": None, "drift_detected": bool(False), "notes": "Test not applicable"})

            # Bar chart for categorical features
            ref_counts = reference_df[feature].value_counts(normalize=True).sort_index()
            cur_counts = current_df[feature].value_counts(normalize=True).sort_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ref_counts.index, y=ref_counts.values, name="Reference", opacity=0.75, marker_color='blue'))
            fig.add_trace(go.Bar(x=cur_counts.index, y=cur_counts.values, name="Current", opacity=0.75, marker_color='red'))
            fig.update_layout(
                barmode="overlay",
                title_text=title,
                xaxis_title_text=feature,
                yaxis_title_text="Percentage",
            )
        figs.append(fig)

        # If drift is detected for this feature, save its plot individually for alerting
        if drift_detected:
            plot_path = Path(plots_dir) / f"drift_plot_{feature}.html"
            fig.write_html(str(plot_path), include_plotlyjs="cdn")
    return figs, drift_results

def create_auc_comparison_plot(auc_scores: dict[str, float]) -> go.Figure:
    """Creates a bar chart comparing AUC across different data splits."""
    fig = go.Figure(
        go.Bar(
            x=list(auc_scores.keys()),
            y=list(auc_scores.values()),
            text=[f"{v:.4f}" for v in auc_scores.values()],
            textposition="auto",
        )
    )
    fig.update_layout(
        title_text="AUC Performance by Data Split",
        yaxis_title_text="AUC Score",
        yaxis=dict(range=[0, 1]),
    )
    return fig

def create_auc_time_trend_plot(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, current_auc: float, target: str, prediction: str
) -> go.Figure:
    """Plots historical AUC from reference data and adds the current AUC."""
    # Automatically find the timestamp column. Prioritize 'feature_snapshot_date'
    # to avoid incorrectly matching other columns like 'Monthly_Inhand_Salary'.
    timestamp_col = "feature_snapshot_date"
    if timestamp_col not in reference_df.columns:
        # Fallback to a broader search if the primary column name is not found
        found_col = None
        for col in reference_df.columns:
            if "date" in col.lower() or "month" in col.lower():
                found_col = col
                break
        timestamp_col = found_col
    if not timestamp_col:
        raise ValueError("Could not automatically determine the timestamp column in the reference data.")

    ref_df = reference_df.copy()
    ref_df[timestamp_col] = pd.to_datetime(ref_df[timestamp_col])
    auc_by_month = (
        ref_df.groupby(pd.Grouper(key=timestamp_col, freq="MS"))
        .apply(lambda x: roc_auc_score(x[target], x[prediction]))
        .reset_index(name="auc")
    )
    auc_by_month = auc_by_month.sort_values(timestamp_col)

    # --- Define data groups and assign each month to a group ---
    train_val_dates = pd.to_datetime(pd.date_range("2023-07-01", "2024-03-01", freq="MS"))
    test_dates = pd.to_datetime(pd.date_range("2024-04-01", "2024-06-01", freq="MS"))
    oot_dates = pd.to_datetime(pd.date_range("2024-07-01", "2024-09-01", freq="MS"))

    def assign_group(dt):
        if dt in train_val_dates:
            return "Train/Validation"
        if dt in test_dates:
            return "Test"
        if dt in oot_dates:
            return "Out-of-Time (OOT)"
        return "Other"

    auc_by_month["group"] = auc_by_month[timestamp_col].apply(assign_group)

    # Derive the timestamp for the current data point by finding the latest
    # date within the current dataframe itself.
    cur_df = current_df.copy()
    cur_df[timestamp_col] = pd.to_datetime(cur_df[timestamp_col])

    auc_by_month_inference = (
    cur_df.groupby(pd.Grouper(key=timestamp_col, freq="MS"))
    .apply(lambda x: roc_auc_score(x[target], x[prediction]))
    .reset_index(name="auc")
    )
    auc_by_month_inference = auc_by_month_inference.sort_values(timestamp_col)

    # --- Create a plot with a separate trace for each group ---
    fig = go.Figure()
    for group_name, group_df in auc_by_month.groupby("group"):
        if group_name == "Other": continue # Skip any months not in the defined groups
        fig.add_trace(
            go.Scatter(
                x=group_df[timestamp_col],
                y=group_df["auc"],
                mode="lines+markers",
                name=group_name,
            )
        )

    # Add the current inference data as a distinct point
    fig.add_trace(
        go.Scatter(
            x=auc_by_month_inference[timestamp_col],
            y=auc_by_month_inference["auc"],
            mode="markers",
            name="Current AUC",
            marker=dict(size=12, symbol="star", color="red"),
        )
    )
    fig.update_layout(
        title_text="AUC Performance Over Time",
        xaxis_title_text="Month",
        yaxis_title_text="AUC Score",
        yaxis=dict(range=[0, 1]),
    )
    return fig


def main() -> int:
    """Main entrypoint for the Plotly report generation script."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("plotly_monitoring")

    logger.info("Loading reference data from %s", args.reference_csv)

    # --- Create output directories upfront to prevent errors ---
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    reference_df = _read_data(args.reference_csv)

    logger.info("Loading current data from %s", args.current_csv)
    current_df = _read_data(args.current_csv)

    logger.info("Loading selected features from %s", args.results_json)
    with Path(args.results_json).open("r", encoding="utf-8") as f:
        results_meta = json.load(f)
    selected_features = results_meta.get("selected_features")
    if not selected_features:
        raise ValueError(f"Could not find 'selected_features' in {args.results_json}")

    # Ensure timestamp column exists where needed
    if args.timestamp_col not in reference_df.columns:
        raise ValueError(f"Timestamp column '{args.timestamp_col}' not found in reference data.")
    if args.timestamp_col not in current_df.columns:
        raise ValueError(f"Timestamp column '{args.timestamp_col}' not found in current data. "
                         "Ensure the inference/evaluation pipeline preserves it.")

    common_features = list(set(selected_features) & set(reference_df.columns) & set(current_df.columns))
    logger.info(f"Found {len(common_features)} common features for monitoring.")

    # --- 1. Feature Drift Analysis ---
    logger.info("Generating feature drift plots...")
    feature_drift_figs, drift_results = create_feature_drift_plots(reference_df, current_df, common_features, args.plots_dir)


    # --- 2. Model Performance (AUC) Analysis ---
    logger.info("Generating AUC comparison plot...")
    # Define date ranges for splits
    reference_df[args.timestamp_col] = pd.to_datetime(reference_df[args.timestamp_col]).apply(lambda dt: dt + relativedelta(months=6))  # add back 6 months offset for features to labels
    train_val_dates = pd.to_datetime(pd.date_range("2023-07-01", "2024-03-01", freq="MS"))
    test_dates = pd.to_datetime(pd.date_range("2024-04-01", "2024-06-01", freq="MS"))
    oot_dates = pd.to_datetime(pd.date_range("2024-07-01", "2024-09-01", freq="MS"))
    logging.info(f"reference_df dates min: {reference_df[args.timestamp_col].min()}, max: {reference_df[args.timestamp_col].max()} ")
    logging.info(f"reference_df unique dates: {reference_df[args.timestamp_col].unique()} ")
    # Filter dataframes and calculate AUC for each split
    auc_scores = {}
    
    df_train_val = reference_df[reference_df[args.timestamp_col].isin(train_val_dates)].copy()
    if not df_train_val.empty:
        auc_scores["Train/Validation"] = roc_auc_score(df_train_val[args.target_col], df_train_val[args.prediction_col])

    df_test = reference_df[reference_df[args.timestamp_col].isin(test_dates)].copy()
    if not df_test.empty:
        auc_scores["Test"] = roc_auc_score(df_test[args.target_col], df_test[args.prediction_col])

    df_oot = reference_df[reference_df[args.timestamp_col].isin(oot_dates)].copy()
    if not df_oot.empty:
        auc_scores["Out-of-Time (OOT)"] = roc_auc_score(df_oot[args.target_col], df_oot[args.prediction_col])

    # Calculate overall AUC for the current period for the trend plot
    cur_auc = roc_auc_score(current_df[args.target_col], current_df[args.prediction_col])

    # For the bar chart, calculate AUC for each individual inference month
    current_df[args.timestamp_col] = pd.to_datetime(current_df[args.timestamp_col])
    inference_auc_by_month = current_df.groupby(pd.Grouper(key=args.timestamp_col, freq='MS')).apply(
        lambda x: roc_auc_score(x[args.target_col], x[args.prediction_col]) if not x.empty else None
    ).dropna()

    for month_timestamp, auc in inference_auc_by_month.items():
        month_str = month_timestamp.strftime('%Y-%m')
        auc_scores[f"{month_str} (Inference)"] = auc

    auc_comp_fig = create_auc_comparison_plot(auc_scores)

    # --- 3. AUC Time Trend Analysis ---
    logger.info("Generating AUC time trend plot...")
    auc_trend_fig = create_auc_time_trend_plot(
        reference_df, current_df, cur_auc, args.target_col, args.prediction_col
    )

    # --- Assemble and save HTML report ---
    logger.info("Assembling HTML report...")
    report_path = Path(args.report_html)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Save individual plots for embedding in alerts ---
    auc_comp_fig.write_html(Path(args.plots_dir) / "auc_comparison_plot.html", include_plotlyjs="cdn")
    auc_trend_fig.write_html(Path(args.plots_dir) / "auc_trend_plot.html", include_plotlyjs="cdn")

    # --- Create Summary Table ---
    logger.info("Creating summary table for HTML report...")
    # 1. Feature Drift Summary
    drifted_features_count = sum(1 for res in drift_results if res.get("drift_detected"))
    drift_summary_text = f"{drifted_features_count} / {len(common_features)} features drifted"

    # 2. Model Performance Status
    is_auc_degraded = False
    oot_auc = auc_scores.get("Out-of-Time (OOT)")
    test_auc = auc_scores.get("Test")
    train_auc = auc_scores.get("Train/Validation")
    inference_aucs = {k: v for k, v in auc_scores.items() if "(Inference)" in k}

    for month, auc in inference_aucs.items():
        if oot_auc and auc < (oot_auc * 0.95):
            is_auc_degraded = True
            break
        if test_auc and auc < (test_auc * 0.93):
            is_auc_degraded = True
            break
        if train_auc and auc < (train_auc * 0.90):
            is_auc_degraded = True
            break
    
    if is_auc_degraded:
        performance_status = "Bad"
        performance_color = "red"
    else:
        performance_status = "Good"
        performance_color = "green"

    summary_table_html = f"""
    <h2>Monitoring Summary</h2>
    <table border="1" style="width:60%; margin-left:auto; margin-right:auto; border-collapse: collapse; text-align: center; font-size: 1.1em;">
      <tr style="background-color: #f2f2f2;">
        <th style="padding: 8px;">Metric</th>
        <th style="padding: 8px;">Status</th>
      </tr>
      <tr>
        <td style="padding: 8px;">Feature Drift</td>
        <td style="padding: 8px;">{drift_summary_text}</td>
      </tr>
      <tr>
        <td style="padding: 8px;">Model AUC Performance</td>
        <td style="padding: 8px; color: {performance_color}; font-weight: bold;">{performance_status}</td>
      </tr>
    </table>
    """

    # --- Assemble and save JSON summary ---
    logger.info("Assembling JSON summary...")
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "auc_scores_by_group": auc_scores,
        "feature_drift_results": drift_results,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with report_path.open("w", encoding="utf-8") as f:
        # Write HTML header
        f.write("""
        <html>
        <head>
            <title>Model Monitoring Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: sans-serif; }
                h1, h2 { color: #333; }
                .plot-container { display: flex; flex-wrap: wrap; justify-content: center; }
                .plot { margin: 15px; border: 1px solid #ddd; box-shadow: 2px 2px 5px #eee; }
            </style>
        </head>
        <body>
            <h1>Model Monitoring Report</h1>
        """)

        # Inject the summary table at the top of the report
        f.write(summary_table_html)

        # Write Model Performance section
        f.write("<h2>Model Performance</h2><div class='plot-container'>")
        f.write(f"<div class='plot'>{auc_comp_fig.to_html(full_html=False, include_plotlyjs=False)}</div>")
        f.write(f"<div class='plot'>{auc_trend_fig.to_html(full_html=False, include_plotlyjs=False)}</div>")
        f.write("</div>")

        # Write Feature Drift section
        f.write("<h2>Feature Distribution Drift</h2><div class='plot-container'>")
        for fig in feature_drift_figs:
            f.write(f"<div class='plot'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>")
        f.write("</div>")

        # Write HTML footer
        f.write("""
        </body>
        </html>
        """)

    logger.info("Successfully saved HTML report to %s", report_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
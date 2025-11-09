#!/usr/bin/env python3
"""
Airflow DAG orchestrating XGBoost inference and evaluation for a monthly snapshot.

Pipeline steps:
1. Run the inference pipeline to score customers and persist predictions.
2. Regenerate labels for the same snapshot and compute evaluation metrics (AUC).

When triggering manually you can override the snapshot month via
``dag_run.conf["snapshot_month"]``. The inference task automatically pulls the
stage-2 model URI from the training results JSON.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
import logging

import pandas as pd
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import pendulum
from airflow.models import Variable
from airflow.utils.email import send_email


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "project"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_JSON = PROJECT_ROOT / "outputs" / "results.json"


def _get_latest_training_run_info(outputs_dir: Path) -> dict:
    """Find the latest training run and return its metadata."""
    model_tuning_dir = outputs_dir / "model_tuning"
    if not model_tuning_dir.exists():
        return {}
    
    latest_run = max(model_tuning_dir.iterdir(), key=lambda p: p.stat().st_mtime if p.is_dir() else 0)
    results_path = latest_run / "results.json"
    
    try:
        data = json.loads(results_path.read_text())
        data['results_json_path'] = str(results_path)
        data['reference_data_path'] = str(latest_run / "reference_data.parquet")
        return data
    except FileNotFoundError:
        return {}

def _resolve_model_path(**kwargs):
    """
    Determines which results.json to use based on DAG run configuration.
    - If use_latest_model is True, find the latest training run.
    - Otherwise, use the custom path provided.
    Pushes the resolved path to XComs.
    """
    conf = kwargs.get("dag_run").conf or {}
    use_latest = conf.get("use_latest_model", True)
    
    if use_latest:
        print("`use_latest_model` is True. Finding the latest model.")
        latest_run_info = _get_latest_training_run_info(PROJECT_ROOT / "outputs")
        if not latest_run_info:
            raise FileNotFoundError("Could not find any training runs in 'outputs/model_tuning'.")
        results_path = latest_run_info["results_json_path"]
        reference_path = latest_run_info["reference_data_path"]
        logging.info(f"Using latest results.json at {results_path}")
        logging.info(f"Using reference data at {reference_path}")
    else:
        print("`use_latest_model` is False. Using custom model path.")
        results_path = conf.get("custom_results_json_path")
        if not results_path or not Path(results_path).exists():
            raise FileNotFoundError(f"Custom results JSON not found or not specified: {results_path}")
        # Infer reference path from the custom results.json path
        reference_path = str(Path(results_path).parent / "reference_data.parquet")
        if not Path(reference_path).exists():
            raise FileNotFoundError(f"Could not find reference_data.parquet alongside custom results JSON: {reference_path}")
    
    kwargs["ti"].xcom_push(key="resolved_results_json", value=str(results_path))
    kwargs["ti"].xcom_push(key="resolved_reference_data_path", value=str(reference_path))

def _generate_months_to_process(**kwargs):
    """
    Generates a list of months to process for dynamic task mapping.
    - For manual runs: If snapshot_month is '2024-12-01', it generates
      ['2024-10-01', '2024-11-01', '2024-12-01'].
    - For scheduled runs: It returns a single month, e.g., ['2025-11-01'].
    """
    dag_run = kwargs["dag_run"]
    is_manual_run = dag_run.run_id.startswith("manual__")

    # For manual runs, get snapshot_month from conf. For scheduled, use the logical date {{ ds }}.
    snapshot_month_str = dag_run.conf.get("snapshot_month", kwargs["params"]["snapshot_month"])

    # The base date from which to start backfilling
    backfill_start_date_str = "2024-10-01"
    backfill_start_date = date.fromisoformat(backfill_start_date_str)
    
    # Clean up potential timestamp from manual trigger UI
    end_date = date.fromisoformat(snapshot_month_str.split('T')[0])

    if is_manual_run and end_date > backfill_start_date:
        # Generate all months from the backfill start date up to the selected end date
        months = pd.date_range(start=backfill_start_date, end=end_date, freq='MS').strftime('%Y-%m-%d').tolist()
        print(f"Manual run detected. Processing months: {months}")
        return months
    else:
        # For scheduled runs or manual runs on/before the start date, process only the single specified month
        print(f"Scheduled run or single month run. Processing: {[snapshot_month_str]}")
        return [snapshot_month_str]

def _aggregate_results(**kwargs):
    """
    Finds all 'inference_with_labels.csv' files from the dynamic tasks,
    concatenates them, and saves the result to a new file.
    """
    ti = kwargs["ti"]
    dag_run = kwargs["dag_run"]
    project_root = Path(kwargs["params"]["project_root"])
    months = ti.xcom_pull(task_ids="generate_months_to_process", key="return_value")
    safe_run_id = dag_run.run_id.replace(':', '_').replace('+', '_').replace('T', '_')
    
    all_dfs = []
    for month in months:
        file_path = project_root / 'reports' / f"{month}_{safe_run_id}" / "inference_with_labels.csv"
        df = pd.read_csv(file_path)
        # Manually add the timestamp column, as it's not preserved by the evaluation script.
        df['feature_snapshot_date'] = month
        all_dfs.append(df)
    concatenated_df = pd.concat(all_dfs, ignore_index=True)
    
    output_path = project_root / 'reports' / f"aggregated_{safe_run_id}" / "aggregated_inference_with_labels.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concatenated_df.to_csv(output_path, index=False)
    return str(output_path)

def _check_and_alert(**kwargs):
    """
    Reads the summary JSON, checks for alerts based on a 3-tier system,
    and saves an HTML file to simulate sending an email if alerts are found.
    """
    # --- 1. Setup and Path Construction ---
    dag_run = kwargs["dag_run"]
    project_root = Path(kwargs["params"]["project_root"])
    safe_run_id = dag_run.run_id.replace(':', '_').replace('+', '_').replace('T', '_')
    output_dir = project_root / 'reports' / f"aggregated_{safe_run_id}"
    summary_path = output_dir / "aggregated_summary.json"
    snapshot_month = dag_run.conf.get("snapshot_month", kwargs["ds"])

    # --- 2. Get Current Run Status ---
    from src.pipelines.check_and_alert import check_alerts
    current_alerts = check_alerts(summary_path)

    # --- 3. Determine Highest Alert Level from Current Run ---
    highest_current_level = "OK"
    levels = [a["level"] for a in current_alerts["auc_degradation_alerts"]]
    if "Critical" in levels: highest_current_level = "Critical"
    elif "Alert" in levels: highest_current_level = "Alert"
    elif "Warning" in levels: highest_current_level = "Warning"

    # --- 4. Generate and Save Alert File ---
    if highest_current_level != "OK" or current_alerts["feature_drift_detected"]:
        subject = f"[{highest_current_level}] Model Monitoring: Issues Detected for {snapshot_month}"
        
        # Build the alert message
        message_parts = []
        if current_alerts["feature_drift_detected"]:
            drifted_features_str = "".join(f"<li>- {res['feature']} (p-value: {res['p_value']:.4f})</li>" for res in current_alerts["drifted_features"])
            message_parts.append(f"<b>Significant feature drift detected:</b><ul>{drifted_features_str}</ul>")
        
        if current_alerts["auc_degradation_alerts"]:
            auc_alerts_str = "".join(f"<li><b>{a['level']}:</b> {a['message']}</li>" for a in current_alerts["auc_degradation_alerts"])
            message_parts.append(f"<b>AUC Performance Degradation:</b><ul>{auc_alerts_str}</ul>")

        # Assemble final content
        html_content = f"""
        <h3>Model Monitoring Alert</h3>
        <p>Issues were detected during the monitoring run for the period ending <b>{snapshot_month}</b>.</p> 
        {''.join(f'<p>{part}</p>' for part in message_parts)}
        <hr>
        """

        # --- Embed Plots if Alerts are Present ---
        plot_html = ""
        if current_alerts["auc_degradation_alerts"]:
            plot_html += """
            <h3>Performance Plots</h3>
            <div style="display: flex; flex-wrap: wrap; justify-content: center;"> 
                <iframe src="data/auc_comparison_plot.html" width="800" height="600" style="border:none;"></iframe>
                <iframe src="data/auc_trend_plot.html" width="800" height="600" style="border:none;"></iframe>
            </div>
            """
        if current_alerts["feature_drift_detected"]:
            plot_html += "<h3>Drifted Feature Plots</h3>"
            for res in current_alerts["drifted_features"]:
                feature_name = res['feature']
                plot_html += f'<iframe src="data/drift_plot_{feature_name}.html" width="800" height="500" style="border:none;"></iframe>'
        
        html_content += plot_html + """
        <p>Please review the monitoring report for details.</p>
        """
        
        # Simulate email by writing to an HTML file
        alert_html_doc = f"""
        <html>
        <head><title>{subject}</title></head>
        <body style="font-family: sans-serif; padding: 20px;">
            {html_content}
        </body>
        </html>
        """
        alert_file_path = output_dir / "alert_simulation.html"
        with open(alert_file_path, "w", encoding="utf-8") as f:
            f.write(alert_html_doc)
        logging.info(f"Alert conditions met. Simulated email saved to: {alert_file_path}")
    else:
        subject = f"[OK] Model Monitoring: All Checks Passed for {snapshot_month}"
        html_content = f"""
        <html>
        <head><title>{subject}</title></head>
        <body style="font-family: sans-serif; padding: 20px;">
            <h3>{subject}</h3>
            <p>All monitoring checks passed. No issues detected.</p>
        </body>
        </html>
        """
        alert_file_path = output_dir / "alert_simulation_OK.html"
        with open(alert_file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info("No alert conditions met. 'All Good' simulation file saved to: %s", alert_file_path)
        
DEFAULT_ARGS = {
    "owner": "ml_analytics",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email": ["your_email@example.com"], # Add your email here
}



with DAG(
    dag_id="xgb_inference_monitoring",
    default_args=DEFAULT_ARGS,
    description="Run monthly XGBoost inference, evaluation, and monitoring. Scheduled for the 1st of each month.",
    # Run at 00:00 on the 1st day of every month.
    schedule="0 0 1 * *",
    # The first run will be on 2025-12-01 for the November 2025 data interval.
    start_date=pendulum.datetime(2025, 11, 1, tz="UTC"),
    catchup=False,
    params={
        "use_latest_model": True,
        "custom_results_json_path": "",
        "project_root": str(PROJECT_ROOT),
        "alert_email": "your_email@example.com", # Default email for alerts
        "threshold": 0.5,
        # Default to the logical date if not provided manually
        # For scheduled runs, this will be {{ ds }}
        "snapshot_month": "{{ ds }}",
    },
    user_defined_macros={
        "pathlib": Path,
    },
    max_active_runs=1,
) as dag:

    generate_months = PythonOperator(
        task_id="generate_months_to_process",
        python_callable=_generate_months_to_process,
        provide_context=True,
    )

    run_inference = BashOperator.partial(
        task_id="run_inference",
        bash_command="""
        {% set results_json_path = ti.xcom_pull(key='resolved_results_json', task_ids='resolve_model_path') %}
        {% set safe_run_id = dag_run.run_id | replace(':', '_') | replace('+', '_') | replace('T', '_') %}
        {% set folder_name = params.month + '_' + safe_run_id %}
        {% set run_reports_dir = pathlib(params.project_root) / 'reports' / folder_name %}
        mkdir -p {{ run_reports_dir }}
        cd {{ params.project_root }} && \
        python src/pipelines/inference_pipeline.py \
          --snapshot-date {{ params.month }} \
          --results-json {{ results_json_path }} \
          --raw-data-root {{ pathlib(params.project_root) / 'data' }} \
          --datamart-root {{ pathlib(params.project_root) / 'datamart' }} \
          --threshold {{ params.threshold }} \
          --output-csv {{ run_reports_dir }}/inference.csv
        """,
    ).expand(params=generate_months.output.map(lambda m: {"month": m}))

    evaluate_predictions = BashOperator.partial(
        task_id="evaluate_predictions",
        bash_command="""
        {% set safe_run_id = dag_run.run_id | replace(':', '_') | replace('+', '_') | replace('T', '_') %}
        {% set folder_name = params.month + '_' + safe_run_id %}
        {% set run_reports_dir = pathlib(params.project_root) / 'reports' / folder_name %}
        cd {{ params.project_root }} && \
        python src/pipelines/evaluation_pipeline.py \
          --snapshot-date {{ params.month }} \
          --inference-csv {{ run_reports_dir }}/inference.csv \
          --datamart-root {{ pathlib(params.project_root) / 'datamart' }}
        """,
    ).expand(params=generate_months.output.map(lambda m: {"month": m}))

    resolve_model_path = PythonOperator(
        task_id="resolve_model_path",
        python_callable=_resolve_model_path,
        provide_context=True,
    )
    
    aggregate_results = PythonOperator(
        task_id="aggregate_results",
        python_callable=_aggregate_results,
        provide_context=True,
    )

    generate_monitoring_report = BashOperator(
        task_id="generate_monitoring_report",
        bash_command="""
        {% set results_json_path = ti.xcom_pull(key='resolved_results_json', task_ids='resolve_model_path') %}
        {% set reference_path = ti.xcom_pull(key='resolved_reference_data_path', task_ids='resolve_model_path') %}
        {% set aggregated_current_csv = ti.xcom_pull(task_ids='aggregate_results', key='return_value') %}
        {% set safe_run_id = dag_run.run_id | replace(':', '_') | replace('+', '_') | replace('T', '_') %}
        {% set output_dir = pathlib(params.project_root) / 'reports' / ('aggregated_' + safe_run_id) %} 
        cd {{ params.project_root }} && \
        python src/pipelines/generate_plotly_report.py \
          --reference-csv {{ reference_path }} \
          --results-json {{ results_json_path }} \
          --current-csv {{ aggregated_current_csv }} \
          --report-html {{ output_dir }}/aggregated_plotly_report.html \
          --plots-dir {{ output_dir }}/data \
          --summary-json {{ output_dir }}/aggregated_summary.json
        """,
    )

    check_and_alert_task = PythonOperator(
        task_id="check_and_alert",
        python_callable=_check_and_alert,
        provide_context=True,
    )

    # Define the new workflow with aggregation
    resolve_model_path >> generate_months >> run_inference >> evaluate_predictions
    evaluate_predictions >> aggregate_results >> generate_monitoring_report >> check_and_alert_task

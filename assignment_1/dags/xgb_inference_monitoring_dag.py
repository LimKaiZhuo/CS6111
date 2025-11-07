#!/usr/bin/env python3
"""
Airflow DAG orchestrating XGBoost inference and evaluation for a monthly snapshot.

Pipeline steps:
1. Run the inference pipeline to score customers and persist predictions.
2. Regenerate labels for the same snapshot and compute evaluation metrics (AUC).

When triggering manually you can override the snapshot month via
``dag_run.conf["snapshot_month"]``. The inference task automatically pulls the
stage-2 model URI (or local model path) from the training results JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "project"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_JSON = PROJECT_ROOT / "outputs" / "results.json"


def _resolve_model_uri(results_path: Path, project_root: Path) -> str:
    """
    Return the best model URI to pass to the inference pipeline.

    Preference order:
    1. Absolute path derived from ``stage2_model_path`` if it exists on disk.
    2. ``stage2_model_artifact_uri`` for MLflow registry URIs.
    """
    try:
        data = json.loads(results_path.read_text())
    except FileNotFoundError:
        return ""

    local_path = data.get("stage2_model_path")
    if local_path:
        abs_path = (project_root / local_path).resolve()
        if abs_path.exists():
            return str(abs_path)

    return data.get("stage2_model_artifact_uri", "")


DEFAULT_ARGS = {
    "owner": "ml_analytics",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}



with DAG(
    dag_id="xgb_inference_monitoring",
    default_args=DEFAULT_ARGS,
    description="Run monthly XGBoost inference, evaluate predictions, and generate Evidently monitoring reports.",
    schedule_interval="@monthly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={
        "project_root": str(PROJECT_ROOT),
        "reports_dir": str(REPORTS_DIR),
        "results_json": str(RESULTS_JSON),
        "data_dir": str(PROJECT_ROOT / "data"),
        "datamart_dir": str(PROJECT_ROOT / "datamart"),
        "threshold": 0.5,
        "snapshot_month": "",
        "model_uri": _resolve_model_uri(RESULTS_JSON, PROJECT_ROOT),
    },
    max_active_runs=1,
) as dag:

    run_inference = BashOperator(
        task_id="run_inference",
        bash_command="""
        {% set snapshot = dag_run.conf.get('snapshot_month', params.snapshot_month or ds) %}
        {% set snapshot_token = snapshot | replace('-', '') %}
        cd {{ params.project_root }} && \
        python {{ params.project_root }}/src/pipelines/inference_pipeline.py \
          --snapshot-date {{ snapshot }} \
          --results-json {{ params.results_json }} \
          --raw-data-root {{ params.data_dir }} \
          --datamart-root {{ params.datamart_dir }} \
          --threshold {{ params.threshold }} \
{% if params.model_uri %}
          --model-uri "{{ params.model_uri }}" \
{% endif %}
          --output-csv {{ params.reports_dir }}/inference_{{ snapshot_token }}.csv
        """,
    )

    evaluate_predictions = BashOperator(
        task_id="evaluate_predictions",
        bash_command="""
        {% set snapshot = dag_run.conf.get('snapshot_month', params.snapshot_month or ds) %}
        {% set snapshot_token = snapshot | replace('-', '') %}
        cd {{ params.project_root }} && \
        python {{ params.project_root }}/src/pipelines/evaluation_pipeline.py \
          --snapshot-date {{ snapshot }} \
          --inference-csv {{ params.reports_dir }}/inference_{{ snapshot_token }}.csv \
          --datamart-root {{ params.datamart_dir }}
        """,
    )

    run_inference >> evaluate_predictions

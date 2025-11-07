#!/usr/bin/env python3
"""
Airflow DAG that backfills bronze/silver/gold data and triggers the XGB training pipeline.

The DAG performs:
1. Bronze/silver/gold backfill for the inclusive range 2023-07-01 .. 2024-09-01.
2. Invokes the Python training pipeline to run the Optuna+SHAP training routine.

All tasks run sequentially using Bash/Python operators to reuse existing scripts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "project"

DEFAULT_ARGS = {
    "owner": "ml_analytics",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

BRONZE_START = "2023-07-01"
BRONZE_END = "2024-09-01"

with DAG(
    dag_id="backfill_train_pipeline",
    default_args=DEFAULT_ARGS,
    description="Backfill feature/label data then run 2 stage hyperparameter tuning and feature selection XGB model.",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={
        "project_root": str(PROJECT_ROOT),
        "bronze_start": BRONZE_START,
        "bronze_end": BRONZE_END,
    },
    max_active_runs=1,
) as dag:

    backfill_data = BashOperator(
        task_id="backfill_data",
        bash_command="""
        cd {{ params.project_root }} && \
        python src/pipelines/backfill_data_pipeline.py \
          --start-date {{ params.bronze_start }} \
          --end-date {{ params.bronze_end }}
        """,
    )

    train_model = BashOperator(
        task_id="run_training_pipeline",
        bash_command="""
        cd {{ params.project_root }} && \
        python src/training/xgb_training_pipeline.py \
          --gold-features-dir {{ params.project_root }}/datamart/gold/feature_store__XGB_v1 \
          --gold-labels-dir {{ params.project_root }}/datamart/gold/label_store \
          --model-save-dir {{ params.project_root }}/models/testing \
          --initial-trials 50 \
          --refine-trials 50 \
          --k-folds 5 \
          --experiment-name 5_fold_feature_select \
          --output-json {{ params.project_root }}/outputs/train_results.json
        """,
    )

    backfill_data >> train_model

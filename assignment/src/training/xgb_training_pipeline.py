#!/usr/bin/env python3
"""
Two-stage Optuna + SHAP tuning pipeline for XGBoost, ready for Airflow.
Splits data, trains twice (with and without SHAP feature groups), and
reports validation / test / OOT metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pyspark.sql
from pyspark.sql import SparkSession
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.data_processing_snapshot_feature_label import (
    gold_features_labels_snapshot_creation,
    train_test_OOT_simulation_months,
)
from src.models.xgb import (
    feature_label_XGB_get_X_y,
    optuna_mlflow_hyperparamter_tuning_xgb,
    shap_explainer_feature_set_generator,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run two-stage XGBoost tuning with SHAP feature grouping and MLflow logging."
    )
    parser.add_argument("--spark-app-name", default="xgb_optuna_dev")
    parser.add_argument("--spark-master", default="local[*]")

    parser.add_argument(
        "--gold-features-dir",
        required=True,
        help="Path to the feature store directory.",
    )
    parser.add_argument(
        "--gold-labels-dir",
        required=True,
        help="Path to the label store directory.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base directory for the 'model_tuning' folder.",
    )
    parser.add_argument("--initial-trials", type=int, default=50)
    parser.add_argument("--refine-trials", type=int, default=50)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--experiment-name", default="xgb_optuna_pipeline")
    parser.add_argument("--mlflow-experiment", default="xgb_optuna")

    return parser


def _evaluate(model, gdf, selected_columns, return_proba=False) -> float:
    X_eval, y_eval = feature_label_XGB_get_X_y(gdf)
    if selected_columns :
        X_eval = X_eval.loc[:,selected_columns ]
    proba = model.predict_proba(X_eval)[:, 1]
    if return_proba:
        return roc_auc_score(y_eval, proba), proba
    else:
        return roc_auc_score(y_eval, proba)


def main() -> int:
    args = _build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("xgb_shap_pipeline")
    logger.info("Arguments: %s", vars(args))

    # Create a timestamped output directory within 'model_tuning'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / "model_tuning" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    spark: SparkSession = (
        SparkSession.builder.appName(args.spark_app_name).master(args.spark_master).getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    mlflow.set_experiment(args.mlflow_experiment)

    logger.info("Splitting dataset into train/test/OOT/sim snapshots")
    train_snapshots, test_snapshots, oot_snapshots = train_test_OOT_simulation_months()
    logging.info("Train snapshots: %s", train_snapshots)
    logging.info("Test snapshots: %s", test_snapshots)
    logging.info("OOT snapshots: %s", oot_snapshots)

    logger.info("Creating gold features/labels for training split")
    gdf_train = gold_features_labels_snapshot_creation(
        spark=spark,
        gold_features_dir=Path(args.gold_features_dir),
        gold_labels_dir=Path(args.gold_labels_dir),
        snapshots=train_snapshots,
    )
    X_train, y_train = feature_label_XGB_get_X_y(gdf_train)

    logger.info("Running stage 1 Optuna tuning (n_trials=%d)", args.initial_trials)
    study_initial, best_model_stage1, model_path_stage1, artifact_uri_stage1 = (
        optuna_mlflow_hyperparamter_tuning_xgb(
            X_train,
            y_train,
            searchspace_dict={"n_trials": args.initial_trials},
            k_folds=args.k_folds,
            save_model_dir=args.output_dir,  # Use the new output directory
            expt_name=f"{args.experiment_name}_stage1",
        )
    )

    feature_groups = shap_explainer_feature_set_generator(
        artifact_uri=artifact_uri_stage1,
        X=X_train,
        y=y_train,
        report_dir=output_dir,  # Save SHAP plots to the new output directory
    )

    logger.info("Running stage 2 Optuna tuning with SHAP feature groups (n_trials=%d)", args.refine_trials)
    (
        study_stage2,
        best_model_stage2,
        model_path_stage2,
        artifact_uri_stage2,
    ) = optuna_mlflow_hyperparamter_tuning_xgb(
        X_train,
        y_train,
        searchspace_dict={"n_trials": args.refine_trials},
        k_folds=args.k_folds,
        save_model_dir=output_dir,  # Save model to the new output directory
        expt_name=f"{args.experiment_name}_stage2",
        feature_set=feature_groups,
    )

    logger.info("Evaluating stage 2 model on test snapshots")
    gdf_test = gold_features_labels_snapshot_creation(
        spark=spark,
        gold_features_dir=Path(args.gold_features_dir),
        gold_labels_dir=Path(args.gold_labels_dir),
        snapshots=test_snapshots,
    )
    selected_columns  = study_stage2.best_trial.params.get("feature_group")
    selected_columns = feature_groups[selected_columns]
    auc_test, proba_test = _evaluate(best_model_stage2, gdf_test, selected_columns, return_proba=True)

    logger.info("Evaluating stage 2 model on OOT snapshots")
    gdf_oot = gold_features_labels_snapshot_creation(
        spark=spark,
        gold_features_dir=Path(args.gold_features_dir),
        gold_labels_dir=Path(args.gold_labels_dir),
        snapshots=oot_snapshots,
    )
    auc_oot, proba_oot = _evaluate(best_model_stage2, gdf_oot, selected_columns, return_proba=True)

    best_trial = study_stage2.best_trial
    selected_group = best_trial.user_attrs.get("feature_group")
    selected_features = feature_groups.get(selected_group) if selected_group else None

    # --- Generate predictions on training data for monitoring reference ---
    logger.info("Generating predictions on training data for monitoring reference.")
    X_train_final = X_train.loc[:, selected_features] if selected_features else X_train
    train_proba = best_model_stage2.predict_proba(X_train_final)[:, 1]

    gdf_all = gold_features_labels_snapshot_creation(
        spark=spark,
        gold_features_dir=Path(args.gold_features_dir),
        gold_labels_dir=Path(args.gold_labels_dir),
        snapshots=train_snapshots + test_snapshots + oot_snapshots,
    )

    # Save reference data for monitoring, now with predictions
    reference_data_path = output_dir / "reference_data.parquet"
    reference_df = gdf_all.toPandas()
    reference_df["default_probability"] = list(train_proba) + list(proba_test) + list(proba_oot)
    reference_df.to_parquet(reference_data_path)
    logger.info("Saved reference data for monitoring to %s", reference_data_path)

    results: dict[str, Any] = {
        "stage1_best_auc": study_initial.best_value,
        "stage1_model_path": str(model_path_stage1),
        "stage1_model_artifact_uri": artifact_uri_stage1,
        "stage2_best_auc": study_stage2.best_value,
        "stage2_model_path": str(model_path_stage2),
        "stage2_model_artifact_uri": artifact_uri_stage2,
        "test_auc": auc_test,
        "reference_data_path": str(reference_data_path),
        "oot_auc": auc_oot,
        "best_params": best_trial.params,
        "selected_feature_group": selected_group,
        "selected_features": selected_features,
        "feature_groups": feature_groups,
    }

    # Save results to the new output directory
    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results written to %s", out_path)

    logger.info(
        "Stage2 AUC (CV)=%0.4f | Test AUC=%0.4f | OOT AUC=%0.4f",
        results["stage2_best_auc"],
        auc_test,
        auc_oot,
    )
    print(json.dumps(results, indent=2))

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python src/training/xgb_training_pipeline.py `
  --gold-features-dir "datamart/gold/feature_store__XGB_v1" `
  --gold-labels-dir "datamart/gold/label_store" `
  --output-dir "outputs" `
  --initial-trials 50 `
  --refine-trials 50 `
  --k-folds 5 `
  --experiment-name "5_fold_feature_select" `
  --output-json ".\outputs\results.json"
  
python src/training/xgb_training_pipeline.py\
  --gold-features-dir datamart/gold/feature_store__XGB_v1 \
  --gold-labels-dir datamart/gold/label_store \
  --output-dir outputs \
  --initial-trials 50 \
  --refine-trials 50 \
  --k-folds 5 \
  --experiment-name 5_fold_feature_select \
  --output-json outputs/results.json


"""

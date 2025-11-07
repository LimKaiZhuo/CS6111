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
        "--model-save-dir",
        required=True,
        help="Directory where tuned models will be persisted.",
    )
    parser.add_argument("--initial-trials", type=int, default=50)
    parser.add_argument("--refine-trials", type=int, default=50)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--experiment-name", default="xgb_optuna_pipeline")
    parser.add_argument("--output-json", help="Optional path to dump final metrics/results JSON.")
    parser.add_argument("--mlflow-experiment", default="xgb_optuna")

    return parser


def _evaluate(model, gdf, selected_columns ) -> float:
    X_eval, y_eval = feature_label_XGB_get_X_y(gdf)
    if selected_columns :
        X_eval = X_eval.loc[:,selected_columns ]
    proba = model.predict_proba(X_eval)[:, 1]
    return roc_auc_score(y_eval, proba)


def main() -> int:
    args = _build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("xgb_shap_pipeline")
    logger.info("Arguments: %s", vars(args))

    spark: SparkSession = (
        SparkSession.builder.appName(args.spark_app_name).master(args.spark_master).getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    mlflow.set_experiment(args.mlflow_experiment)

    logger.info("Splitting dataset into train/test/OOT/sim snapshots")
    train_snapshots, test_snapshots, oot_snapshots = train_test_OOT_simulation_months()

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
            save_model_dir=args.model_save_dir,
            expt_name=f"{args.experiment_name}_stage1",
        )
    )

    logger.info("Generating SHAP feature sets from stage 1 model")
    feature_groups = shap_explainer_feature_set_generator(
        artifact_uri=artifact_uri_stage1,
        X=X_train,
        y=y_train,
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
        save_model_dir=args.model_save_dir,
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
    auc_test = _evaluate(best_model_stage2, gdf_test, selected_columns )

    logger.info("Evaluating stage 2 model on OOT snapshots")
    gdf_oot = gold_features_labels_snapshot_creation(
        spark=spark,
        gold_features_dir=Path(args.gold_features_dir),
        gold_labels_dir=Path(args.gold_labels_dir),
        snapshots=oot_snapshots,
    )
    auc_oot = _evaluate(best_model_stage2, gdf_oot, selected_columns )

    best_trial = study_stage2.best_trial
    selected_group = best_trial.user_attrs.get("feature_group")
    selected_features = feature_groups.get(selected_group) if selected_group else None

    results: dict[str, Any] = {
        "stage1_best_auc": study_initial.best_value,
        "stage1_model_path": str(model_path_stage1),
        "stage1_model_artifact_uri": artifact_uri_stage1,
        "stage2_best_auc": study_stage2.best_value,
        "stage2_model_path": str(model_path_stage2),
        "stage2_model_artifact_uri": artifact_uri_stage2,
        "test_auc": auc_test,
        "oot_auc": auc_oot,
        "best_params": best_trial.params,
        "selected_feature_group": selected_group,
        "selected_features": selected_features,
        "feature_groups": feature_groups,
    }

    output_json = args.output_json
    if output_json:
        out_path = Path(output_json)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if out_path.suffix:
            out_path = out_path.with_name(f"{out_path.stem}_{timestamp}{out_path.suffix}")
        else:
            out_path = out_path / f"results_{timestamp}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
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
  --model-save-dir "models/testing" `
  --initial-trials 50 `
  --refine-trials 50 `
  --k-folds 5 `
  --experiment-name "5_fold_feature_select" `
  --output-json ".\outputs\results.json"
  
python src/training/xgb_training_pipeline.py\
  --gold-features-dir datamart/gold/feature_store__XGB_v1 \
  --gold-labels-dir datamart/gold/label_store \
  --model-save-dir models/testing \
  --initial-trials 50 \
  --refine-trials 50 \
  --k-folds 5 \
  --experiment-name 5_fold_feature_select \
  --output-json outputs/results.json


"""

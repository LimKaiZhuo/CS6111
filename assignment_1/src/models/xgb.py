import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Optional
from datetime import datetime
import pandas as pd
import xgboost as xgb
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import json
from pathlib import Path
import ast
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold


def feature_label_XGB_get_X_y(features_labels: SparkDataFrame):
    if not isinstance(features_labels, SparkDataFrame):
        raise TypeError("features_labels must be a pyspark.sql.DataFrame")

    pdf = features_labels.toPandas()
    if pdf.empty:
        raise ValueError("features_labels DataFrame is empty; nothing to train on.")

    ignore_cols = {
        "label",
        "loan_id",
        "Customer_ID",
        "snapshot_date", 
        "feature_snapshot_date"
    }

    candidate_cols = [col for col in pdf.columns if col not in ignore_cols]
    if not candidate_cols:
        raise ValueError("No candidate feature columns detected.")

    numeric_cols = []
    categorical_cols = []
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(pdf[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns found (numeric or categorical).")

    print(f'Data has {len(numeric_cols)} numeric cols and {len(categorical_cols)} cat cols. Data shape is {pdf.shape}')

    # Keep only data needed for training and drop rows missing any required values.
    working_df = pdf[["label", *numeric_cols, *categorical_cols]].dropna()
    dropped = len(pdf) - len(working_df)
    if dropped:
        warnings.warn(
            f"Dropped {dropped} rows with missing label or feature values before training.",
            RuntimeWarning,
        )

    if working_df.empty:
        raise ValueError("All rows were dropped during NA cleanup; cannot train model.")

    X = working_df.drop(columns=["label"]).copy()
    y = working_df["label"].astype("int32")
    return X,y


def feature_label_XGB_training(
    spark: SparkSession,
    features_labels: SparkDataFrame,
    hparams: Dict[str, Any],
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Train an XGBoost classifier on the joined gold feature/label snapshot data.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session (kept for signature parity; not used directly).
    features_labels : pyspark.sql.DataFrame
        Spark DataFrame containing a `label` column plus feature columns. Identifier fields
        such as `loan_id`, `Customer_ID`, or snapshot markers are ignored during training.
        Non-numeric columns are treated as categorical features.
    hparams : dict
        Hyperparameters forwarded to `xgboost.XGBClassifier`. `enable_categorical` defaults
        to True (and `tree_method` to "hist") unless supplied explicitly.

    Returns
    -------
    xgboost.XGBClassifier
        Fitted XGBoost classifier.
    dict
        Training metadata (feature names, label name, row count, snapshot counts).
    """
    if not isinstance(features_labels, SparkDataFrame):
        raise TypeError("features_labels must be a pyspark.sql.DataFrame")

    pdf = features_labels.toPandas()
    if pdf.empty:
        raise ValueError("features_labels DataFrame is empty; nothing to train on.")

    ignore_cols = {
        "label",
        "loan_id",
        "Customer_ID",
        "snapshot_date", 
        "feature_snapshot_date"
    }

    candidate_cols = [col for col in pdf.columns if col not in ignore_cols]
    if not candidate_cols:
        raise ValueError("No candidate feature columns detected.")

    numeric_cols = []
    categorical_cols = []
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(pdf[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns found (numeric or categorical).")

    print(f'Data has {len(numeric_cols)} numeric cols and {len(categorical_cols)} cat cols.')

    # Keep only data needed for training and drop rows missing any required values.
    working_df = pdf[["label", *numeric_cols, *categorical_cols]].dropna()
    dropped = len(pdf) - len(working_df)
    if dropped:
        warnings.warn(
            f"Dropped {dropped} rows with missing label or feature values before training.",
            RuntimeWarning,
        )

    if working_df.empty:
        raise ValueError("All rows were dropped during NA cleanup; cannot train model.")

    X = working_df.drop(columns=["label"]).copy()
    y = working_df["label"].astype("int32")

    # Convert categorical columns to pandas category dtype.
    for col in categorical_cols:
        X[col] = X[col].astype("category")

    # Ensure numeric columns are floating point (optional but common for XGB).
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Turn on native categorical handling unless caller already set it.
    model_kwargs = dict(hparams or {})
    model_kwargs.setdefault("enable_categorical", True)
    model_kwargs.setdefault("tree_method", "hist")

    model = xgb.XGBClassifier(**model_kwargs)
    model.fit(X, y)

    metadata = {
        "feature_names": list(X.columns),
        "label_name": "label",
        "row_count": len(working_df),
        "feature_snapshot_count": (
            pdf["feature_snapshot_date"].nunique()
            if "feature_snapshot_date" in pdf
            else None
        ),
        "label_snapshot_count": (
            pdf["label_snapshot_date"].nunique()
            if "label_snapshot_date" in pdf
            else None
        ),
    }

    return model, metadata



def optuna_mlflow_hyperparamter_tuning_xgb(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | Iterable[int],
    searchspace_dict: Dict[str, Callable[[optuna.trial.Trial], Any]],
    k_folds: int,
    save_model_dir: str | Path,
    expt_name: str ,
    feature_set: Optional[dict[str, list[str]]] = None,
) -> Tuple[optuna.study.Study, xgb.XGBClassifier, Path]:
    """
    Run Optuna hyperparameter search for an XGBoost classifier, log every trial to MLflow,
    and persist the best model.

    Parameters
    ----------
    X : pandas.DataFrame | numpy.ndarray
        Feature matrix. Columns with pandas ``category`` dtype are passed to XGBoost as
        categorical features; everything else is treated as numeric.
    y : pandas.Series | numpy.ndarray | Iterable[int]
        Binary labels aligned with ``X``.
    searchspace_dict : dict[str, Callable[[optuna.trial.Trial], Any]]
        Mapping from parameter names to callables that draw values via Optuna’s
        ``trial.suggest_*`` APIs. A special key ``"_n_trials"`` (optional) controls the
        number of optimisation trials.
    k_folds : int
        Number of Stratified K-Fold splits used to evaluate each trial.
    save_model_dir : str | pathlib.Path
        Directory where the best model will be saved in XGBoost JSON format.

    Returns
    -------
    optuna.study.Study
        Completed Optuna study with all trial results.
    xgboost.XGBClassifier
        Classifier refit on the full dataset using the best-found hyperparameters.
    pathlib.Path
        Filesystem path to the saved best-model artifact.
    """
    if not isinstance(searchspace_dict, dict) or not searchspace_dict:
        raise ValueError("searchspace_dict must be a non-empty dict of trial callables.")

    save_dir = Path(save_model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_trials = searchspace_dict.get("n_trials", 50)
    


    X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
    y_series = pd.Series(y).reset_index(drop=True)

    if len(X_df) != len(y_series):
        raise ValueError("X and y must contain the same number of rows.")

    data = pd.concat([X_df.reset_index(drop=True), y_series.rename("label")], axis=1).dropna()
    dropped = len(X_df) - len(data)
    if dropped:
        warnings.warn(f"Dropped {dropped} rows with missing values before tuning.", RuntimeWarning)

    X_df = data.drop(columns="label")
    y_series = data["label"].astype("int32")

    categorical_cols = [
        col for col in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[col])
    ]
    for col in categorical_cols:
        X_df[col] = X_df[col].astype("category")

    for col in X_df.columns:
        if col not in categorical_cols:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    mlflow.set_experiment("xgb_optuna")

    feature_keys = list(feature_set.keys()) if feature_set else None

    def _build_params(trial: optuna.Trial) -> Dict[str, Any]:
        params = {
        # learning capacity / complexity
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),  # minimum split loss

        # sampling / column subsampling (regularises interaction structure)
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

        # L1 / L2-style penalties
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 20.0, log=True),
        }
    
        params.setdefault("tree_method", "hist")
        params.setdefault("enable_categorical", True)
        params.setdefault("eval_metric", "auc")
        return params

    def objective(trial: optuna.Trial) -> float:
        auc_scores, acc_scores, prec_scores, rec_scores, f1_scores, logloss_scores = (
            [], [], [], [], [], []
        )

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            params = _build_params(trial)
            if feature_keys:
                feature_key = trial.suggest_categorical("feature_group", feature_keys)
                selected_columns = feature_set[feature_key]
                if not selected_columns:
                    raise optuna.TrialPruned(f"Feature group '{feature_key}' is empty.")
                X_trial = X_df.loc[:, selected_columns]
                trial.set_user_attr("feature_group", feature_key)
            else:
                X_trial = X_df

            mlflow.log_params(params)
            print(f'X_trial has shape {X_trial.shape}')
            for train_idx, valid_idx in skf.split(X_trial, y_series):
                X_train, X_valid = X_trial.iloc[train_idx], X_trial.iloc[valid_idx]
                y_train, y_valid = y_series.iloc[train_idx], y_series.iloc[valid_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, verbose=False)

                proba = model.predict_proba(X_valid)[:, 1]
                preds = (proba >= 0.5).astype(int)

                auc_scores.append(roc_auc_score(y_valid, proba))
                acc_scores.append(accuracy_score(y_valid, preds))
                prec_scores.append(precision_score(y_valid, preds, zero_division=0))
                rec_scores.append(recall_score(y_valid, preds, zero_division=0))
                f1_scores.append(f1_score(y_valid, preds, zero_division=0))
                logloss_scores.append(log_loss(y_valid, proba, labels=[0, 1]))

            mean_auc = float(np.mean(auc_scores))
            mlflow.log_metric("mean_auc", mean_auc)
            mlflow.log_metric("mean_accuracy", float(np.mean(acc_scores)))
            mlflow.log_metric("mean_precision", float(np.mean(prec_scores)))
            mlflow.log_metric("mean_recall", float(np.mean(rec_scores)))
            mlflow.log_metric("mean_f1", float(np.mean(f1_scores)))
            mlflow.log_metric("mean_logloss", float(np.mean(logloss_scores)))
            return mean_auc
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    expt_name = f"{timestamp}_{expt_name}"

    with mlflow.start_run(run_name=expt_name):
        study = optuna.create_study(study_name=expt_name, direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = _build_params(study.best_trial)
        best_model = xgb.XGBClassifier(**best_params)

        final_features = X_df
        if feature_keys:
            feature_key = study.best_trial.user_attrs.get("feature_group") if study.best_trial.user_attrs else None
            if feature_key is None:
                feature_key = study.best_trial.params.get("feature_group")
            if feature_key is None:
                raise KeyError(
                    "Feature-set optimisation was enabled but the winning trial has no recorded feature group."
                )
            selected_columns = feature_set[feature_key]
            final_features = X_df.loc[:, selected_columns]
            print(f"final model feature frame shape is {final_features.shape} using group '{feature_key}'")

        best_model.fit(final_features, y_series)

        model_path = save_dir / f"xgb_optuna_best_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        best_model.save_model(model_path.as_posix())

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_auc", study.best_value)
        model_info = mlflow.xgboost.log_model(
            best_model,
            artifact_path="best_model",
            registered_model_name=expt_name  # only if you really want registry
        )
        mlflow.log_artifact(model_path.as_posix())

        best_model_artifact_uri =  model_info.model_uri

    return study, best_model, model_path, best_model_artifact_uri



def knee_from_shap(shap_totals: pd.Series):
    """Return knee index, feature name, selected features, and cumulative share."""
    # Ensure descending order
    shap_totals = shap_totals.sort_values(ascending=False)

    cumprop = shap_totals.cumsum() / shap_totals.sum()
    straight_line = np.linspace(1 / len(shap_totals), 1.0, len(shap_totals))
    deviation = cumprop.values - straight_line

    knee_pos = int(np.argmax(deviation))
    knee_feature = shap_totals.index[knee_pos]
    selected_features = shap_totals.index[:knee_pos + 1]
    knee_share = float(cumprop.iloc[knee_pos])

    return knee_pos, knee_feature, selected_features, knee_share

def bin_cutting(shap_totals: pd.Series): 
    total = shap_totals.sum()
    cumprop = shap_totals.cumsum() / total

    edges = np.array([0, 0.05 ,0.1,0.2,0.3,0.4,0.5,0.7,0.9,1.1])

    labels = [f"{int(edges[i]*100)}–{int(edges[i+1]*100)}%" for i in range(len(edges)-1)]
    bin_assignments = pd.cut(cumprop, bins=edges, labels=labels, include_lowest=True)

    # list features per bin (only bins that actually get members appear)
    feature_bins = {
        label: shap_totals[bin_assignments == label].index.tolist()
        for label in bin_assignments.cat.categories
        if (bin_assignments == label).any()
    }

    keys = list(feature_bins.keys())
    for idx, key in enumerate(keys):
        if idx == 0:
            continue
        feature_bins[key] = feature_bins[keys[idx-1]] + feature_bins[key]

    pop_groups = []
    for idx, (key, features) in enumerate(feature_bins.items()):
        group_total = shap_totals[features].sum()/total
        if len(features)<3:
            pop_groups.append(key)
            print(f'DROPPING bin {idx+1}: {len(features)} features with {group_total:.4f} ---- due to <3 features in group')
        else:
            print(f'bin {idx+1}: {len(features)} features with {group_total:.4f}')
    for key in pop_groups:
        feature_bins.pop(key)

    return feature_bins


def shap_explainer_feature_set_generator(artifact_uri, X, y,):
    """
    1) take model, use features X and labels y to generate SHAP values
    2) calculate abs SHAP values total for each feature
    3) generate feature set A which is N bins of features cut by cummulative total
    4) generate feature set B which is elbow point based on straight line method
    5) generate SHAP plots and save to mlflow artifact_uri path
    6) return feature sets based on column names

    Args:
        model (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_
    """

    loaded = mlflow.xgboost.load_model(artifact_uri)
    booster = loaded if isinstance(loaded, xgb.Booster) else loaded.get_booster()
    dmatrix = xgb.DMatrix(X, label=y, enable_categorical=True)

    # --- compute SHAP values -----------------------------------------------------------
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(dmatrix)
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_totals = shap_df.abs().sum(axis=0).sort_values(ascending=False)

    # Feature set A: 
    feature_set = bin_cutting(shap_totals)

    # Feature set B: using straight line method
    knee_pos, knee_feature, selected_features, knee_share = knee_from_shap(shap_totals)

    # combine feature set
    feature_set['knee'] = list(selected_features)

    model_id = artifact_uri.split('models:/')[1]
    report_dir = Path(f'reports/{model_id}')
    report_dir.mkdir(parents=True,exist_ok=True)
    # Plotting
    # Determine where to write artifacts

    fig,axes = plt.subplots(nrows=2, ncols=1)
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    axes[0].set_title("Global Feature Importance (SHAP)")

    plt.sca(axes[1])
    shap.summary_plot(shap_values, X, show=False)
    axes[1].set_title("Feature Impact Distribution (SHAP)")
    fig.tight_layout()
    fig.savefig(report_dir / "shap_summary.png", dpi=180)

    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_totals = shap_df.abs().sum(axis=0).sort_values(ascending=False)

    fig,ax = plt.subplots(figsize=(15,10))
    sns.barplot(x=shap_totals.index, y=shap_totals.values, color="steelblue", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
    fig.tight_layout()
    fig.savefig(report_dir / "shap_all.png", dpi=180)

    target = report_dir / "feature_sets.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(feature_set, indent=2), encoding="utf-8")
    
    return feature_set




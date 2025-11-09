from pathlib import Path
from functools import reduce
from dateutil.relativedelta import relativedelta
import warnings
import pandas as pd

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

def train_test_OOT_simulation_months():
    train = pd.date_range("2023-07-01", "2024-03-01", freq="MS").strftime("%Y-%m-%d").tolist()
    test = pd.date_range("2024-04-01", "2024-06-01", freq="MS").strftime("%Y-%m-%d").tolist()
    oot = pd.date_range("2024-07-01", "2024-09-01", freq="MS").strftime("%Y-%m-%d").tolist()
    #sim = pd.date_range("2024-10-01", "2024-12-01", freq="MS").strftime("%Y-%m-%d").tolist()
    return train, test, oot


def gold_features_labels_snapshot_creation(
    spark: SparkSession,
    gold_features_dir,
    gold_labels_dir,
    snapshots,
) -> DataFrame:
    """
    Build a training set by pairing each label snapshot with the feature snapshot taken six
    months earlier.

    For every snapshot identifier in `snapshots` the function:
        1. Loads the corresponding parquet file from `gold_labels_dir`.
        2. Loads the lagged feature parquet (snapshot minus six months) from `gold_features_dir`.
        3. Merges labels and features on the shared identifiers.
        4. Drops rows with missing feature columns, emitting a warning when that happens.
        5. only keep the label column from the label table

    The per-snapshot Spark DataFrames are unioned row-wise and returned.

    Args:
        spark: Active SparkSession.
        gold_features_dir (str | Path): Directory holding gold feature parquet files.
        gold_labels_dir (str | Path): Directory holding gold label parquet files.
        snapshots (Iterable[str | datetime-like]): Snapshot identifiers (e.g., "2024-03-01").

    Returns:
        pyspark.sql.DataFrame: Combined features-plus-labels dataset across all requested snapshots.
    """
    gold_features_dir = Path(gold_features_dir)
    gold_labels_dir = Path(gold_labels_dir)

    if isinstance(snapshots, (str, Path)):
        snapshots = [snapshots]

    merged_frames: list[DataFrame] = []

    for snapshot in snapshots:
        snapshot_py_ts = pd.Timestamp(snapshot)

        label_path = _resolve_snapshot_path(
            gold_labels_dir, "gold_label_store", snapshot_py_ts
        )
        features_snapshot_ts = snapshot_py_ts - relativedelta(months=6)
        feature_path = _resolve_snapshot_path(
            gold_features_dir, "gold_feature_store", features_snapshot_ts
        )

        labels_df = spark.read.parquet(label_path).withColumn(
            "label_snapshot_date", F.lit(snapshot_py_ts.date().isoformat())
        )

        features_df = spark.read.parquet(feature_path)
        if "snapshot_date" in features_df.columns:
            features_df = features_df.drop("snapshot_date")
        features_df = features_df.withColumn(
            "feature_snapshot_date", F.lit(features_snapshot_ts.date().isoformat())
        )

        join_keys = [
            key
            for key in ["Customer_ID"]
            if key in labels_df.columns and key in features_df.columns
        ]
        if not join_keys:
            raise ValueError(
                f"No shared join key between label snapshot {label_path} "
                f"and feature snapshot {feature_path}."
                f"FEATURE COLUMNS {features_df.columns}."
                f"LABEL COLUMNS {labels_df.columns}."
            )
        
        if "label" not in labels_df.columns:
            raise ValueError(f"'label' column not found in {label_path}")
        
        labels_df = labels_df.select(*(join_keys + ["label"]))  # only keep label and join_keys

        merged = labels_df.join(features_df, on=join_keys, how="left")

        feature_columns = [c for c in features_df.columns if c not in join_keys]
        if feature_columns:
            before = merged.count()
            merged = merged.dropna("any", subset=feature_columns)
            dropped = before - merged.count()
            if dropped:
                warnings.warn(
                    f"{dropped} rows dropped for snapshot {snapshot_py_ts.date()} due to missing features.",
                    RuntimeWarning,
                )

        merged_frames.append(merged)

    if not merged_frames:
        return spark.createDataFrame([], schema=spark.read.parquet(label_path).schema)

    return reduce(lambda left, right: left.unionByName(right, allowMissingColumns=True), merged_frames)


def _resolve_snapshot_path(base_dir: Path, prefix: str, snapshot_ts: pd.Timestamp) -> str:
    """
    Map a snapshot timestamp to a parquet path.

    Accepts both single parquet files (prefix_YYYY_MM_DD.parquet) and Spark-style
    directories that may or may not retain the .parquet suffix.
    """
    snapshot_str = snapshot_ts.strftime("%Y_%m_%d")
    candidate = base_dir / f"{prefix}_{snapshot_str}.parquet"

    if candidate.exists():
        return str(candidate)
    stripped = candidate.with_suffix("")
    if stripped.exists():
        return str(stripped)

    matches = list(base_dir.glob(f"*{snapshot_str}*.parquet"))
    if matches:
        return str(matches[0])

    raise FileNotFoundError(f"No parquet found for {prefix} {snapshot_str} in {base_dir}")

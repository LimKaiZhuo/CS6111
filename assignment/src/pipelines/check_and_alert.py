#!/usr/bin/env python3
"""
Reads a monitoring summary JSON and checks for drift or performance degradation.

This script is designed to be called by an Airflow task to determine if an alert
should be sent. It checks for two main conditions:
1. Any feature has statistically significant drift.
2. The AUC of any inference month drops below a set percentage of the historical
   Train, Test, or OOT AUCs.

The script prints alerts to stdout and exits with a non-zero status code if
any alert conditions are met.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check monitoring summary for alerts.")
    parser.add_argument("--summary-json", required=True, help="Path to the aggregated_summary.json file.")
    return parser


def check_alerts(summary_path: Path) -> dict:
    """
    Parses the summary JSON and returns a list of alert messages if any
    conditions are met.

    Args:
        summary_path: Path to the summary JSON file.

    Returns:
        A dictionary containing structured alert information for feature drift
        and AUC degradation.
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"Monitoring summary not found at {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    alert_results = {
        "feature_drift_detected": False,
        "drifted_features": [],
        "auc_degradation_alerts": [],
    }

    # --- 1. Check for Feature Drift ---
    drift_results = summary.get("feature_drift_results", [])
    alert_results["drifted_features"] = [res for res in drift_results if res.get("drift_detected")]
    if alert_results["drifted_features"]:
        alert_results["feature_drift_detected"] = True

    # --- 2. Check for AUC Degradation ---
    auc_scores = summary.get("auc_scores_by_group", {})
    oot_auc = auc_scores.get("Out-of-Time (OOT)")

    inference_aucs = {k: v for k, v in auc_scores.items() if "(Inference)" in k}

    if oot_auc:
        for month_str, auc in inference_aucs.items():
            degradation_level = None
            message = ""
            if auc < (oot_auc * 0.85):  # 15% drop
                degradation_level = "Critical"
                message = f"{month_str} AUC ({auc:.4f}) is >15% below OOT AUC ({oot_auc:.4f}). Model retraining may be required."
            elif auc < (oot_auc * 0.90):  # 10% drop
                degradation_level = "Alert"
                message = f"{month_str} AUC ({auc:.4f}) is >10% below OOT AUC ({oot_auc:.4f}). Diagnostic investigation is recommended."
            elif auc < (oot_auc * 0.95):  # 5% drop
                degradation_level = "Warning"
                message = f"{month_str} AUC ({auc:.4f}) is >5% below OOT AUC ({oot_auc:.4f}). Closer observation is advised."

            if degradation_level:
                alert_results["auc_degradation_alerts"].append({
                    "month": month_str,
                    "level": degradation_level,
                    "message": message,
                })
    else:
        logging.warning("OOT AUC not found in summary. Cannot check for AUC degradation.")

    return alert_results


def main() -> int:
    """Main entrypoint for the alert checking script."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("alert_checker")

    try:
        alerts = check_alerts(Path(args.summary_json))
        if alerts:
            logger.warning("Alert conditions met!")
            for alert in alerts:
                # Print alerts for logging purposes
                print(alert.replace("<b>", "").replace("</b>", "").replace("<ul>", "\n").replace("</ul>", "").replace("<li>", "  - ").replace("</li>", ""))
            # Exit with a non-zero status code to indicate an alert state
            return 1
        else:
            logger.info("No alert conditions met. All checks passed.")
            return 0
    except Exception as e:
        logger.error("An error occurred during alert checking: %s", e, exc_info=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
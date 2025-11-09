# CS6111 Assignment: XGBoost Loan Default Prediction & Monitoring Pipeline

This project is a complete MLOps pipeline for a loan default prediction model. It handles everything from preparing the data and training the model to running monthly predictions and monitoring for any issues. The whole process is automated with Apache Airflow, running inside Docker containers.

## Key Features

*   **Data Pipeline**: A PySpark job prepares all the historical feature and label data needed for training.
*   **Model Training with Hyperparameter Tuning**: The training pipeline uses Optuna to find the best hyperparameters in a two-stage process and MLflow to track experiments. It also uses SHAP to find the best groups of features.
*   **Monthly Inference Simulation**: A monthly Airflow job simulates a real-world scenario where the model makes predictions on new data.
*   **Custom Monitoring Reports**: After inference, a custom report is built with Plotly to check for problems. It analyzes:
    *   **Feature Drift**: Checks if the new data's distribution is different from the training data using statistical tests (K-S test and Chi-squared).
    *   **Model Performance Drift**: Tracks the model's AUC score over time and compares it to how it performed on the original training, test, and out-of-time datasets.
*   **Automated Alerts**: A 3-tier alert system (Warning, Alert, Critical) checks if the model's performance has dropped. If it has, it generates a detailed HTML alert file with embedded plots so you can see what's wrong right away.

## Technology Stack

*   **Orchestration**: Apache Airflow 
*   **Containerization**: Docker, Docker Compose
*   **Data Processing**: PySpark, Pandas
*   **Machine Learning**: XGBoost, Scikit-learn
*   **Hyperparameter Tuning**: Optuna
*   **Model Explainability**: SHAP
*   **Experiment Tracking**: MLflow
*   **Visualization**: Plotly


---

## How to Run the Pipelines

This project is set up to run with Docker.

### 1. Setup and Installation

1.  **What you need**: Make sure you have Docker and Docker Compose installed.
2.  **Build and Run**: Go to the `docker/airflow` directory in your terminal and run this command. It will build the Docker image and start up Airflow.
    ```bash
    docker-compose up --build
    ```
3.  **Check out the Airflow UI**: Open your web browser and go to `http://localhost:8080`. You should see the Airflow dashboard with the DAGs for this project.

### 2. Running the Training Pipeline

The `backfill_train_pipeline` DAG prepares all the data and then trains a new model.

1.  In the Airflow UI, find the `backfill_train_pipeline` DAG and enable it.
2.  Click the "Play" button on the right to trigger a new DAG run.
3.  You can customize the run using the "Trigger DAG w/ config" option. Key parameters include:
    *   `skip_backfill`: Set to `true` to skip the data generation steps if the data already exists.
    *   `stage1_trials`, `stage2_trials`: Number of Optuna trials for hyperparameter tuning.
    *   `k_folds`: Number of folds for cross-validation during training.

Upon completion, this pipeline will generate a new timestamped model directory inside `assignment/outputs/model_tuning/`. This directory contains the trained model, `results.json` with performance metrics, and `reference_data.parquet` for monitoring.

### 3. Running the Inference and Monitoring Pipeline

The `xgb_inference_monitoring` DAG simulates a monthly production job that scores new data and monitors the model's health.

1.  In the Airflow UI, find the `xgb_inference_monitoring` DAG and enable it.
2.  Click the "Play" button and choose "Trigger DAG w/ config" to run it.
3.  Set the inference snapshot month to 2024-12-01 for a demonstration (it will run 2024-10-01, 2024-11-01, and 2024-12-01).


#### Key Configuration Parameters:

*   **`snapshot_month`**: This is the most important parameter. It specifies the end of the period to process.
    *   **Single Month Run**: To run for just one month (e.g., October 2024), set it to `"2024-10-01"`.
    *   **Backfill Run**: To run for a range of months, set it to the end of the range. For example, setting it to `"2024-12-01"` will automatically trigger parallel inference and evaluation tasks for October, November, and December 2024, then aggregate the results into a single report.
*   **`use_latest_model`**: If `true` (default), the DAG will automatically find and use the most recently trained model from the `outputs/model_tuning/` directory.
*   **`alert_email`**: The email address where the simulated alert should be sent.

---

## Output Artifacts

After running the monitoring pipeline, check the `assignment/reports/` directory. A new folder named `aggregated_<run_id>` will be created for your run.

Inside this folder, you will find:

*   **`aggregated_plotly_report.html`**: This is the main report you'll want to look at. It has a summary table at the top, charts for AUC performance, and plots for feature drift. Any features that have drifted will have a **bold, red title**.
*   **`aggregated_summary.json`**: A JSON file with all the raw numbers from the analysis, like AUC scores and the p-values from the drift tests. This is useful for any further automated processing.
*   **`alert_simulation.html`** (or `alert_simulation_OK.html`): A fake email alert (as an HTML file). If there are problems, this file will summarize them and show you the specific charts for the AUC degradation or feature drift so you can see what's wrong right away.
*   **`data/`** (subdirectory): This folder contains all the supporting files used to build the reports:
    *   Individual HTML files for each plot that appears in the alert.
    *   The combined inference data (`aggregated_inference_with_labels.csv`).

## Directory Structure

```
assignment/
├── dags/                  # Airflow DAG definitions
├── datamart/              # Processed data layers (bronze, silver, gold)
├── docker/                # Docker and environment configuration
├── outputs/               # Outputs from the training pipeline (models, results)
├── reports/               # Outputs from the monitoring pipeline (HTML reports, alerts)
├── src/                   # Source code for all pipelines, models, and utilities
│   ├── pipelines/
│   ├── training/
│   └── models/
└── utils/                 # Shared utility functions
```

```
"""
Example Apache Airflow DAG for AutoML Pipeline

This file demonstrates how to integrate the automl_airflow_tasks.py functions
into a production Apache Airflow DAG.

To use this DAG:
1. Copy this file to your Airflow DAGs folder (usually ~/airflow/dags/)
2. Ensure automl_airflow_tasks.py is in the Python path or in the same directory
3. Configure the DAG parameters below
4. Enable the DAG in the Airflow UI

Author: MLOps Engineering Team
Date: 2025-12-01
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import os

# Import the task functions from our module
from automl_airflow_tasks import (
    check_new_data,
    run_autogluon_search,
    deploy_best_model
)

# =============================================================================
# DAG CONFIGURATION
# =============================================================================

# Default arguments for all tasks in the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['mlops@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# Configuration variables
DATA_DIRECTORY = os.getenv('AUTOML_DATA_DIR', '/data/input')
MODEL_OUTPUT_DIR = os.getenv('AUTOML_MODEL_DIR', '/models/autogluon')
DEPLOYMENT_TARGET = os.getenv('DEPLOYMENT_TARGET', 'production')

# =============================================================================
# DAG DEFINITION
# =============================================================================

dag = DAG(
    dag_id='automl_pipeline',
    default_args=default_args,
    description='End-to-end AutoML pipeline with AutoGluon and Dask',
    schedule_interval=timedelta(hours=6),  # Run every 6 hours
    catchup=False,
    max_active_runs=1,  # Only one pipeline run at a time
    tags=['automl', 'autogluon', 'dask', 'ml-pipeline', 'production']
)

# =============================================================================
# TASK 1: CHECK FOR NEW DATA (SENSOR)
# =============================================================================

def check_data_callable(**context):
    """Wrapper function for the data check sensor"""
    file_path = check_new_data(
        data_directory=DATA_DIRECTORY,
        file_pattern='*.csv'
    )
    
    if file_path:
        # Push the file path to XCom for downstream tasks
        context['ti'].xcom_push(key='data_file_path', value=file_path)
        return True
    return False


check_data_task = PythonSensor(
    task_id='check_new_data',
    python_callable=check_data_callable,
    poke_interval=300,  # Check every 5 minutes
    timeout=3600,  # Timeout after 1 hour
    mode='poke',  # 'poke' mode blocks the worker, 'reschedule' frees it
    soft_fail=False,  # Fail the task if timeout is reached
    dag=dag,
    doc_md="""
    ### Check for New Data
    
    This sensor task monitors the data directory for new CSV files.
    
    **Behavior:**
    - Checks every 5 minutes (poke_interval)
    - Times out after 1 hour if no new data arrives
    - Passes the file path to downstream tasks via XCom
    
    **Configuration:**
    - Data Directory: `{}`
    - File Pattern: `*.csv`
    """.format(DATA_DIRECTORY)
)

# =============================================================================
# TASK 2: RUN AUTOGLUON SEARCH WITH DASK
# =============================================================================

def run_training_callable(**context):
    """Wrapper function for AutoGluon training"""
    ti = context['ti']
    
    # Pull the data file path from upstream task
    train_file = ti.xcom_pull(task_ids='check_new_data', key='data_file_path')
    
    if not train_file:
        raise ValueError("No training file path received from upstream task")
    
    # Run AutoGluon search
    model_results = run_autogluon_search(
        train_file=train_file,
        test_file=None,  # Test file is optional
        target_column='target',
        time_limit=600,  # 10 minutes training time
        preset_quality='good_quality',  # Options: best_quality, high_quality, good_quality, medium_quality
        num_bag_folds=5,  # K-fold bagging for ensemble
        num_stack_levels=1,  # Stacking levels
        enable_dask=True,
        dask_scheduler='threads',  # Options: 'threads', 'processes', or cluster address
        output_dir=MODEL_OUTPUT_DIR
    )
    
    # Push results to XCom for deployment task
    ti.xcom_push(key='model_results', value=model_results)
    
    return model_results


autogluon_task = PythonOperator(
    task_id='run_autogluon_search',
    python_callable=run_training_callable,
    provide_context=True,
    # Resource configuration for Kubernetes executor
    executor_config={
        'KubernetesExecutor': {
            'request_memory': '8Gi',
            'request_cpu': '4',
            'limit_memory': '16Gi',
            'limit_cpu': '8',
            'node_selector': {
                'node-type': 'ml-training'
            }
        }
    },
    pool='ml_training_pool',  # Use a specific pool for resource management
    priority_weight=10,  # Higher priority than other tasks
    dag=dag,
    doc_md="""
    ### Run AutoGluon Search
    
    This task performs the core ML training using AutoGluon with Dask integration.
    
    **Process:**
    1. Loads training data from the file path provided by upstream task
    2. Initializes AutoGluon TabularPredictor
    3. Configures Dask for distributed computing
    4. Runs hyperparameter search with ensemble methods
    5. Generates model leaderboard
    
    **Key Parameters:**
    - Training Time: 10 minutes (600 seconds)
    - Quality Preset: good_quality
    - Bag Folds: 5 (K-fold bagging)
    - Stack Levels: 1 (stacking layers)
    - Dask: Enabled with thread scheduler
    
    **Resources:**
    - CPU: 4-8 cores
    - Memory: 8-16 GB
    - Pool: ml_training_pool
    """
)

# =============================================================================
# TASK 3: DEPLOY BEST MODEL
# =============================================================================

def deploy_model_callable(**context):
    """Wrapper function for model deployment"""
    ti = context['ti']
    
    # Pull model results from upstream task
    model_results = ti.xcom_pull(task_ids='run_autogluon_search', key='model_results')
    
    if not model_results:
        raise ValueError("No model results received from upstream task")
    
    # Deploy the best model
    deployment_info = deploy_best_model(
        model_results=model_results,
        deployment_target=DEPLOYMENT_TARGET
    )
    
    # Push deployment info for monitoring/logging
    ti.xcom_push(key='deployment_info', value=deployment_info)
    
    return deployment_info


deploy_task = PythonOperator(
    task_id='deploy_best_model',
    python_callable=deploy_model_callable,
    provide_context=True,
    dag=dag,
    doc_md="""
    ### Deploy Best Model
    
    This task deploys the best performing model to the target environment.
    
    **Process:**
    1. Retrieves model results from upstream task
    2. Loads the trained AutoGluon predictor
    3. Analyzes the leaderboard to find the best model
    4. Packages model artifacts
    5. Deploys to the target environment
    6. Registers model in model registry (simulated)
    
    **Deployment Target:**
    - Environment: `{}`
    
    **Post-Deployment:**
    - Model endpoint is activated
    - Monitoring dashboards are updated
    - Team notifications are sent
    """.format(DEPLOYMENT_TARGET)
)

# =============================================================================
# TASK DEPENDENCIES
# =============================================================================

# Define the task execution order
check_data_task >> autogluon_task >> deploy_task

# =============================================================================
# DAG DOCUMENTATION
# =============================================================================

dag.doc_md = """
# AutoML Pipeline with AutoGluon and Dask

This DAG implements an end-to-end automated machine learning pipeline.

## Overview

The pipeline consists of three main tasks:

1. **check_new_data**: Monitors for new datasets
2. **run_autogluon_search**: Trains models using AutoGluon with Dask
3. **deploy_best_model**: Deploys the best model to production

## Schedule

- **Frequency**: Every 6 hours
- **Start Date**: Yesterday
- **Max Active Runs**: 1 (prevents concurrent executions)

## Resource Requirements

- **Training Task**: 4-8 CPU cores, 8-16 GB RAM
- **Other Tasks**: Standard resources

## Monitoring

- Email notifications on failure
- Model metrics logged via XCom
- Grafana dashboards for resource monitoring

## Configuration

Environment variables:
- `AUTOML_DATA_DIR`: Input data directory (default: /data/input)
- `AUTOML_MODEL_DIR`: Model output directory (default: /models/autogluon)
- `DEPLOYMENT_TARGET`: Deployment environment (default: production)

## Notes

- Ensure Dask cluster is configured before running
- Models are automatically versioned by timestamp
- Failed runs will retry twice with 5-minute delays
"""

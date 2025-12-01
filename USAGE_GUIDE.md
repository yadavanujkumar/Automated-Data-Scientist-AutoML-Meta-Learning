# AutoML Pipeline Usage Guide

This guide provides detailed instructions for using the AutoML Airflow pipeline with AutoGluon and Dask.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Detailed Usage](#detailed-usage)
3. [Configuration Options](#configuration-options)
4. [Airflow Integration](#airflow-integration)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yadavanujkumar/Automated-Data-Scientist-AutoML-Meta-Learning.git
cd Automated-Data-Scientist-AutoML-Meta-Learning

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Run complete pipeline with default settings
python automl_airflow_tasks.py

# Or with custom parameters
python automl_airflow_tasks.py --time_limit 600 --data_dir ./my_data
```

## Detailed Usage

### Task 1: Data Simulation

Generate synthetic training and test datasets:

```python
from automl_airflow_tasks import generate_dummy_dataset

# Generate 1000 samples
train_file, test_file = generate_dummy_dataset(
    output_dir="./data",
    n_samples=1000,
    random_state=42
)

print(f"Training data: {train_file}")
print(f"Test data: {test_file}")
```

**Output:**
```
Generated training dataset: ./data/train.csv with 800 samples
Generated test dataset: ./data/test.csv with 200 samples
```

### Task 2: Check for New Data

Monitor a directory for new CSV files:

```python
from automl_airflow_tasks import check_new_data

# Check for new CSV files
latest_file = check_new_data(
    data_directory="./data",
    file_pattern="*.csv"
)

if latest_file:
    print(f"Found new data: {latest_file}")
else:
    print("No new data found")
```

### Task 3: Run AutoGluon Search

Train models with AutoGluon and Dask:

```python
from automl_airflow_tasks import run_autogluon_search

# Run AutoGluon training
model_results = run_autogluon_search(
    train_file="./data/train.csv",
    test_file="./data/test.csv",
    target_column="target",
    time_limit=600,  # 10 minutes
    preset_quality="good_quality",
    num_bag_folds=5,
    num_stack_levels=1,
    enable_dask=True,
    dask_scheduler="threads",
    output_dir="./models"
)

print(f"Best model: {model_results['best_model']}")
print(f"Model path: {model_results['model_path']}")
```

**Key Parameters:**

- **time_limit**: Training time budget in seconds
  - 60-300: Quick experiments
  - 300-600: Good quality models
  - 600+: Best quality models

- **preset_quality**: Model quality vs speed tradeoff
  - `medium_quality`: Fast, moderate accuracy
  - `good_quality`: Balanced (recommended)
  - `high_quality`: Better accuracy, slower
  - `best_quality`: Best accuracy, slowest

- **num_bag_folds**: K-fold bagging for ensembling
  - 2-3: Fast, less robust
  - 5-8: Balanced (recommended)
  - 10+: Very robust, much slower

- **num_stack_levels**: Stacked ensembling layers
  - 0: No stacking (fastest)
  - 1: Single level (recommended)
  - 2-3: Multiple levels (slower, better)

### Task 4: Deploy Best Model

Deploy the trained model:

```python
from automl_airflow_tasks import deploy_best_model

# Deploy model
deployment_info = deploy_best_model(
    model_results=model_results,
    deployment_target="production"
)

print(f"Deployed: {deployment_info['model_name']}")
print(f"Accuracy: {deployment_info['score_percentage']:.2f}%")
print(f"Status: {deployment_info['status']}")
```

## Configuration Options

### Environment Variables

Set these environment variables for production deployment:

```bash
export AUTOML_DATA_DIR="/data/input"
export AUTOML_MODEL_DIR="/models/autogluon"
export DEPLOYMENT_TARGET="production"
```

### Dask Configuration

#### Local Threading (Default)

```python
# Uses threads in the same process
model_results = run_autogluon_search(
    ...,
    enable_dask=True,
    dask_scheduler="threads"
)
```

#### Local Multiprocessing

```python
# Uses separate processes
model_results = run_autogluon_search(
    ...,
    enable_dask=True,
    dask_scheduler="processes"
)
```

#### Distributed Cluster

```python
from dask.distributed import Client

# Connect to existing Dask cluster
client = Client("scheduler-address:8786")

model_results = run_autogluon_search(
    ...,
    enable_dask=True,
    dask_scheduler="scheduler-address:8786"
)
```

### AutoGluon Advanced Configuration

For more control over model training:

```python
# In run_autogluon_search function, modify the fit call:

predictor.fit(
    train_data=train_data,
    time_limit=time_limit,
    presets=preset_quality,
    
    # Ensemble configuration
    num_bag_folds=5,
    num_stack_levels=1,
    
    # Model-specific hyperparameters
    hyperparameters={
        'GBM': {},  # LightGBM models
        'CAT': {},  # CatBoost models
        'XGB': {},  # XGBoost models
        'NN_TORCH': {},  # Neural Network models
        'RF': {},  # Random Forest models
    },
    
    # Resource allocation
    num_cpus='auto',
    num_gpus=0,  # Set to 1+ for GPU training
    
    # Advanced options
    ag_args_fit={
        'num_cpus': 'auto',
        'num_gpus': 0,
    }
)
```

## Airflow Integration

### Step 1: Copy Files to Airflow

```bash
# Copy DAG to Airflow directory
cp example_airflow_dag.py ~/airflow/dags/

# Copy tasks module
cp automl_airflow_tasks.py ~/airflow/dags/
```

### Step 2: Configure Airflow Pools

Create a resource pool for ML training:

```bash
# Using Airflow CLI
airflow pools set ml_training_pool 2 "Pool for ML training tasks"
```

Or via Airflow UI:
1. Navigate to Admin > Pools
2. Create new pool: `ml_training_pool` with slots=2

### Step 3: Enable the DAG

```bash
# List DAGs
airflow dags list

# Enable the DAG
airflow dags unpause automl_pipeline
```

### Step 4: Trigger the DAG

```bash
# Trigger a DAG run
airflow dags trigger automl_pipeline

# Or schedule it to run automatically (configured in DAG)
# The DAG runs every 6 hours by default
```

### Step 5: Monitor Execution

- Access Airflow UI: http://localhost:8080
- View DAG graph and task status
- Check logs for each task
- Monitor XCom values passed between tasks

## Troubleshooting

### Issue: "No module named 'autogluon'"

**Solution:**
```bash
pip install autogluon
```

### Issue: "Insufficient memory"

**Solutions:**

1. Reduce `num_bag_folds`:
```python
num_bag_folds=3  # Instead of 5
```

2. Lower quality preset:
```python
preset_quality="medium_quality"  # Instead of best_quality
```

3. Reduce training time:
```python
time_limit=300  # 5 minutes instead of 10
```

### Issue: "Dask scheduler not responding"

**Solutions:**

1. Use local threading:
```python
dask_scheduler="threads"
```

2. Restart Dask cluster:
```bash
dask-scheduler --host 0.0.0.0 --port 8786
dask-worker scheduler-address:8786
```

### Issue: "File not found error"

**Solution:**

Ensure data directory exists:
```python
import os
os.makedirs("./data", exist_ok=True)
```

### Issue: "Training takes too long"

**Solutions:**

1. Reduce time limit:
```python
time_limit=60  # 1 minute for testing
```

2. Use faster preset:
```python
preset_quality="medium_quality"
```

3. Disable stacking:
```python
num_stack_levels=0
```

## Best Practices

### 1. Data Preparation

- Clean data before training
- Handle missing values appropriately
- Ensure proper train/test split
- Validate data quality

### 2. Model Training

- Start with shorter time limits for testing
- Gradually increase resources as needed
- Use cross-validation (bag_folds) for robust models
- Monitor resource usage

### 3. Production Deployment

- Version all models with timestamps
- Keep model artifacts in centralized storage (S3, GCS)
- Implement model monitoring and drift detection
- Set up automated retraining pipelines
- Use proper CI/CD for deployment

### 4. Resource Management

- Use Airflow pools to limit concurrent training
- Configure appropriate executor resources
- Monitor memory and CPU usage
- Scale Dask cluster based on workload

### 5. Monitoring

- Log all model metrics
- Track data quality metrics
- Set up alerts for failures
- Monitor model performance in production
- Create dashboards for visibility

### 6. Security

- Don't commit credentials to version control
- Use secure credential management (Airflow connections)
- Implement proper access controls
- Validate all input data
- Scan for security vulnerabilities

## Example Workflows

### Development Workflow

```bash
# 1. Generate test data
python automl_airflow_tasks.py --mode generate_data

# 2. Run quick test (1 minute)
python automl_airflow_tasks.py --time_limit 60

# 3. Review results
# Check ./autogluon_models/ for trained models
```

### Production Workflow

```bash
# 1. Deploy to Airflow
cp example_airflow_dag.py ~/airflow/dags/

# 2. Configure environment
export AUTOML_DATA_DIR="/data/production"
export DEPLOYMENT_TARGET="production"

# 3. Enable DAG
airflow dags unpause automl_pipeline

# 4. Monitor in Airflow UI
# Access http://localhost:8080
```

### Custom Model Training

```python
from automl_airflow_tasks import run_autogluon_search

# Custom configuration for specific use case
model_results = run_autogluon_search(
    train_file="./data/train.csv",
    target_column="target",
    time_limit=1800,  # 30 minutes
    preset_quality="best_quality",
    num_bag_folds=8,
    num_stack_levels=2,
    enable_dask=True,
    output_dir="./models/production"
)
```

## Additional Resources

- [AutoGluon Documentation](https://auto.gluon.ai/)
- [Dask Documentation](https://docs.dask.org/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Project Repository](https://github.com/yadavanujkumar/Automated-Data-Scientist-AutoML-Meta-Learning)

## Support

For issues and questions:
1. Check this usage guide
2. Review the main README.md
3. Check existing GitHub issues
4. Open a new issue with detailed description

# Quick Reference Guide

## Getting Started (30 seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline
python automl_airflow_tasks.py
```

## Core Functions

### Generate Data
```python
from automl_airflow_tasks import generate_dummy_dataset
train_file, test_file = generate_dummy_dataset(output_dir="./data", n_samples=1000)
```

### Check for New Data
```python
from automl_airflow_tasks import check_new_data
latest_file = check_new_data(data_directory="./data")
```

### Train Models
```python
from automl_airflow_tasks import run_autogluon_search
results = run_autogluon_search(
    train_file="./data/train.csv",
    time_limit=300,
    num_bag_folds=5,
    enable_dask=True
)
```

### Deploy Model
```python
from automl_airflow_tasks import deploy_best_model
deployment = deploy_best_model(results, deployment_target='production')
```

## Key Parameters

### Training Parameters
- `time_limit`: 60-300 (quick), 300-600 (good), 600+ (best)
- `preset_quality`: medium_quality, good_quality, high_quality, best_quality
- `num_bag_folds`: 2-3 (fast), 5-8 (balanced), 10+ (robust)
- `num_stack_levels`: 0 (none), 1 (recommended), 2-3 (advanced)

### Dask Configuration
- `enable_dask=True`: Enable distributed computing
- `dask_scheduler="threads"`: Local threading (default)
- `dask_scheduler="processes"`: Local multiprocessing
- `dask_scheduler="address:port"`: Remote cluster

## Airflow Integration

```bash
# Copy to Airflow
cp example_airflow_dag.py ~/airflow/dags/
cp automl_airflow_tasks.py ~/airflow/dags/

# Enable DAG
airflow dags unpause automl_pipeline

# Trigger run
airflow dags trigger automl_pipeline
```

## File Structure

```
automl_airflow_tasks.py      # Main implementation (747 lines)
example_airflow_dag.py        # Airflow DAG (295 lines)
test_automl_airflow.py        # Tests (265 lines)
requirements.txt              # Dependencies
README.md                     # Full documentation
USAGE_GUIDE.md               # Detailed guide
IMPLEMENTATION_SUMMARY.md    # Technical details
VALIDATION_REPORT.md         # Validation results
```

## Common Commands

```bash
# Generate data only
python automl_airflow_tasks.py --mode generate_data

# Show DAG structure
python automl_airflow_tasks.py --mode show_dag

# Custom training time
python automl_airflow_tasks.py --time_limit 600

# Run tests
pytest test_automl_airflow.py -v
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `num_bag_folds` or `time_limit` |
| Too slow | Use `medium_quality` preset |
| Import errors | `pip install -r requirements.txt` |

## Documentation

- **Quick Start**: This file
- **Usage Guide**: USAGE_GUIDE.md
- **API Reference**: Docstrings in automl_airflow_tasks.py
- **Architecture**: README.md

## Support

- GitHub Issues: [Create Issue](https://github.com/yadavanujkumar/Automated-Data-Scientist-AutoML-Meta-Learning/issues)
- Documentation: README.md and USAGE_GUIDE.md

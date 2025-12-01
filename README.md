# Automated Data Scientist - AutoML with Meta-Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-1.0.0-green)](https://auto.gluon.ai/)
[![Dask](https://img.shields.io/badge/Dask-Distributed-orange)](https://dask.org/)
[![Airflow](https://img.shields.io/badge/Airflow-Conceptual-red)](https://airflow.apache.org/)

## 🚀 Overview

This project provides a **production-ready blueprint** for automating end-to-end Machine Learning workflows using **AutoGluon** for AutoML and **Dask** for distributed computing. The system is designed to work with **Apache Airflow** for orchestration, enabling scalable ML pipelines that can handle any tabular dataset.

### Key Features

- 🤖 **Automated Model Selection**: AutoGluon automatically trains and selects the best model
- 🔧 **Distributed Computing**: Dask integration for scalable hyperparameter search
- 📊 **Production Ready**: Complete Airflow DAG blueprint with three core tasks
- 🎯 **Ensemble Learning**: Bagging and stacking for robust model performance
- 📈 **Comprehensive Logging**: Detailed execution logs and model leaderboards
- 🔄 **End-to-End Pipeline**: From data ingestion to model deployment

## 🏗️ Architecture

The pipeline consists of three main Airflow tasks:

```
┌─────────────────────┐
│  check_new_data     │  (Task 1: Python Sensor)
│  Monitor CSV files  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ run_autogluon_search│  (Task 2: Python Operator)
│ Train with Dask     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ deploy_best_model   │  (Task 3: Python Operator)
│ Deploy via Registry │
└─────────────────────┘
```

### Task 1: Check New Data (Sensor)
- Monitors a directory for new CSV files
- Acts as a trigger for the pipeline
- Returns path to the latest dataset

### Task 2: Run AutoGluon Search (Core ML Task)
- Loads training data
- Configures AutoGluon TabularPredictor
- Integrates with Dask for distributed computing
- **Key Parameters**:
  - `num_bag_folds`: K-fold bagging for ensemble learning (default: 5)
  - `num_stack_levels`: Stacked ensembling layers (default: 1)
  - `num_gpus`: GPU allocation for distributed training
  - `time_limit`: Training time budget
- Generates comprehensive model leaderboard

### Task 3: Deploy Best Model
- Analyzes model leaderboard
- Selects highest-scoring model
- Simulates deployment to production
- Registers model in model registry

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yadavanujkumar/Automated-Data-Scientist-AutoML-Meta-Learning.git
cd Automated-Data-Scientist-AutoML-Meta-Learning

# Install dependencies
pip install -r requirements.txt
```

### Optional: Install Apache Airflow

```bash
# For Airflow orchestration (optional)
pip install apache-airflow==2.8.0
pip install apache-airflow-providers-dask==1.1.0
```

## 🎯 Usage

### Standalone Execution (Without Airflow)

```bash
# Run the complete pipeline with default settings
python automl_airflow_tasks.py

# Generate only dummy data
python automl_airflow_tasks.py --mode generate_data

# Show DAG structure
python automl_airflow_tasks.py --mode show_dag

# Run with custom time limit (600 seconds)
python automl_airflow_tasks.py --time_limit 600
```

### Python API Usage

```python
from automl_airflow_tasks import (
    generate_dummy_dataset,
    check_new_data,
    run_autogluon_search,
    deploy_best_model
)

# Step 1: Generate or use existing data
train_file, test_file = generate_dummy_dataset(output_dir="./data")

# Step 2: Check for new data
latest_file = check_new_data(data_directory="./data")

# Step 3: Run AutoGluon search
model_results = run_autogluon_search(
    train_file=train_file,
    test_file=test_file,
    time_limit=300,
    num_bag_folds=5,
    enable_dask=True
)

# Step 4: Deploy best model
deployment_info = deploy_best_model(
    model_results=model_results,
    deployment_target='production'
)

print(f"Deployed: {deployment_info['model_name']}")
print(f"Accuracy: {deployment_info['score_percentage']:.2f}%")
```

### Integration with Apache Airflow

The code includes a complete conceptual DAG structure. To deploy in Airflow:

1. Copy `automl_airflow_tasks.py` to your Airflow DAGs folder
2. Create a DAG file (example provided in the `create_automl_dag()` function)
3. Configure Dask cluster connection
4. Set up resource pools and executor configs
5. Enable the DAG in Airflow UI

## 🔧 Configuration

### AutoGluon Parameters

Customize model training by adjusting these parameters in `run_autogluon_search()`:

```python
predictor.fit(
    time_limit=600,              # Training time budget (seconds)
    presets='best_quality',      # Quality preset: best_quality, high_quality, medium_quality
    num_bag_folds=5,             # K-fold bagging (higher = better, slower)
    num_stack_levels=2,          # Stacking layers (0-3)
    num_cpus='auto',             # CPU allocation
    num_gpus=0,                  # GPU allocation (set >0 for GPU training)
)
```

### Dask Configuration

For distributed computing, configure Dask cluster:

```python
from dask.distributed import Client

# Local cluster
client = Client(n_workers=4, threads_per_worker=2)

# Or connect to existing cluster
client = Client('scheduler-address:8786')
```

## 📊 Output and Results

### Model Leaderboard Example

```
                      model  score_val  pred_time_val   fit_time  
0     WeightedEnsemble_L2    0.9250        0.125        120.45
1     LightGBM_BAG_L1       0.9180        0.089         45.23
2     CatBoost_BAG_L1       0.9165        0.102         52.18
3     XGBoost_BAG_L1        0.9155        0.095         48.76
```

### Deployment Output

```
🚀 DEPLOYMENT SUCCESSFUL: WeightedEnsemble_L2
   Accuracy: 92.50%
   Target: production
   Timestamp: 2025-12-01 17:38:45
```

## 🎓 Technical Details

### Dask Integration Points

1. **Data Loading**: Use `dask.dataframe` for large datasets
2. **Model Training**: AutoGluon distributes training across Dask workers
3. **Hyperparameter Search**: Parallel evaluation of configurations
4. **Resource Scaling**: Dynamic worker allocation

### Ensemble Methods

- **Bagging**: K-fold cross-validation with `num_bag_folds`
- **Stacking**: Multi-level ensembles with `num_stack_levels`
- **Model Diversity**: Multiple algorithm families (LightGBM, XGBoost, CatBoost, Neural Networks)

### AutoML Workflow

1. **Data Preprocessing**: Automatic feature engineering and encoding
2. **Model Selection**: Trains multiple algorithm types
3. **Hyperparameter Tuning**: Bayesian optimization
4. **Ensemble Creation**: Weighted ensemble of top models
5. **Validation**: Cross-validation and holdout testing

## 📝 Project Structure

```
Automated-Data-Scientist-AutoML-Meta-Learning/
├── automl_airflow_tasks.py    # Main pipeline implementation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Generated datasets (created at runtime)
└── autogluon_models/           # Trained models (created at runtime)
```

## 🔒 Security Considerations

- Do not commit model artifacts or datasets to version control
- Use secure credential management for model registry access
- Implement proper access controls for deployment endpoints
- Validate input data before training
- Monitor model performance and data drift

## 🚀 Production Deployment

### Recommended Infrastructure

- **Orchestration**: Apache Airflow on Kubernetes
- **Compute**: Dask distributed cluster (multi-node)
- **Storage**: S3/GCS for datasets and model artifacts
- **Model Registry**: MLflow or SageMaker Model Registry
- **Monitoring**: Grafana + Prometheus for metrics

### Resource Requirements

- **Minimum**: 4 CPU cores, 8 GB RAM
- **Recommended**: 8+ CPU cores, 16+ GB RAM
- **GPU Training**: NVIDIA GPU with CUDA support (optional)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 👥 Authors

- MLOps Engineering Team

## 📚 References

- [AutoGluon Documentation](https://auto.gluon.ai/)
- [Dask Documentation](https://docs.dask.org/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Open a new issue with detailed description

---

**Note**: This is a conceptual blueprint designed for educational and production use. Adapt the configuration based on your specific requirements and infrastructure.
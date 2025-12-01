"""
Test Suite for AutoML Airflow Tasks

This module contains tests for the AutoML pipeline functions.
Tests validate the basic functionality without requiring heavy ML training.

Run tests with: pytest test_automl_airflow.py
"""

import pytest
import os
import tempfile
import shutil
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Import functions to test
from automl_airflow_tasks import (
    generate_dummy_dataset,
    check_new_data,
    create_automl_dag
)


class TestDataGeneration:
    """Tests for data generation functions"""
    
    def test_generate_dummy_dataset(self):
        """Test that dummy dataset generation works correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate dataset
            train_file, test_file = generate_dummy_dataset(
                output_dir=tmpdir,
                n_samples=100,
                random_state=42
            )
            
            # Verify files exist
            assert os.path.exists(train_file)
            assert os.path.exists(test_file)
            
            # Load and verify data
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            
            # Check shapes
            assert len(train_df) == 80  # 80% of 100
            assert len(test_df) == 20   # 20% of 100
            
            # Check columns
            assert 'target' in train_df.columns
            assert 'target' in test_df.columns
            assert len(train_df.columns) == 21  # 20 features + 1 target
    
    def test_generate_dummy_dataset_custom_size(self):
        """Test dataset generation with custom size"""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_file, test_file = generate_dummy_dataset(
                output_dir=tmpdir,
                n_samples=500,
                random_state=123
            )
            
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            
            assert len(train_df) == 400  # 80% of 500
            assert len(test_df) == 100   # 20% of 500


class TestDataSensor:
    """Tests for data checking/sensor functionality"""
    
    def test_check_new_data_file_exists(self):
        """Test that check_new_data finds existing CSV files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test CSV file
            test_file = os.path.join(tmpdir, 'test.csv')
            pd.DataFrame({'col1': [1, 2, 3]}).to_csv(test_file, index=False)
            
            # Check for new data
            result = check_new_data(data_directory=tmpdir)
            
            # Verify result
            assert result is not None
            assert result == test_file
    
    def test_check_new_data_no_files(self):
        """Test that check_new_data returns None when no files exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_new_data(data_directory=tmpdir)
            assert result is None
    
    def test_check_new_data_multiple_files(self):
        """Test that check_new_data returns the latest file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files with different timestamps
            import time
            
            file1 = os.path.join(tmpdir, 'file1.csv')
            pd.DataFrame({'col1': [1]}).to_csv(file1, index=False)
            time.sleep(0.1)
            
            file2 = os.path.join(tmpdir, 'file2.csv')
            pd.DataFrame({'col1': [2]}).to_csv(file2, index=False)
            
            # Check for new data
            result = check_new_data(data_directory=tmpdir)
            
            # Should return the most recent file
            assert result == file2
    
    def test_check_new_data_directory_not_exists(self):
        """Test behavior when directory doesn't exist"""
        result = check_new_data(data_directory='/nonexistent/directory')
        assert result is None


class TestDAGStructure:
    """Tests for DAG structure and documentation"""
    
    def test_create_automl_dag(self):
        """Test that DAG structure can be created"""
        dag_structure = create_automl_dag()
        
        # Verify it returns a string with DAG information
        assert isinstance(dag_structure, str)
        assert 'check_new_data' in dag_structure
        assert 'run_autogluon_search' in dag_structure
        assert 'deploy_best_model' in dag_structure
        assert 'DASK' in dag_structure


class TestAutogluonSearch:
    """Tests for AutoGluon search functionality"""
    
    @patch('automl_airflow_tasks.TabularPredictor')
    def test_run_autogluon_search_basic(self, mock_predictor):
        """Test basic AutoGluon search flow with mocked predictor"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            train_file = os.path.join(tmpdir, 'train.csv')
            test_data = pd.DataFrame({
                'feature_1': [1, 2, 3, 4, 5],
                'feature_2': [5, 4, 3, 2, 1],
                'target': [0, 1, 0, 1, 0]
            })
            test_data.to_csv(train_file, index=False)
            
            # Mock the predictor
            mock_instance = MagicMock()
            mock_instance.get_model_best.return_value = 'BestModel'
            mock_instance.leaderboard.return_value = pd.DataFrame({
                'model': ['BestModel'],
                'score_val': [0.95]
            })
            mock_predictor.return_value = mock_instance
            
            # Import and test (this will use the mock)
            from automl_airflow_tasks import run_autogluon_search
            
            results = run_autogluon_search(
                train_file=train_file,
                time_limit=10,
                output_dir=tmpdir
            )
            
            # Verify results structure
            assert 'model_path' in results
            assert 'best_model' in results
            assert 'leaderboard' in results
            assert results['dask_enabled'] == True


class TestDeployment:
    """Tests for model deployment functionality"""
    
    @patch('automl_airflow_tasks.TabularPredictor')
    def test_deploy_best_model_basic(self, mock_predictor):
        """Test basic deployment flow with mocked predictor"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock model results
            model_results = {
                'model_path': tmpdir,
                'best_model': 'TestModel',
                'leaderboard': {'model': ['TestModel'], 'score_val': [0.92]},
                'test_score': 0.91
            }
            
            # Mock the predictor
            mock_instance = MagicMock()
            mock_instance.leaderboard.return_value = pd.DataFrame({
                'model': ['TestModel', 'Model2'],
                'score_val': [0.92, 0.88]
            })
            mock_instance.info.return_value = {'model_info': {}}
            mock_predictor.load.return_value = mock_instance
            
            # Import and test
            from automl_airflow_tasks import deploy_best_model
            
            deployment_info = deploy_best_model(
                model_results=model_results,
                deployment_target='staging'
            )
            
            # Verify deployment info
            assert deployment_info['status'] == 'success'
            assert deployment_info['deployment_target'] == 'staging'
            assert 'model_name' in deployment_info
            assert 'score_percentage' in deployment_info


class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_full_pipeline_structure(self):
        """Test that all components can be imported and initialized"""
        # This test verifies that the module structure is correct
        from automl_airflow_tasks import (
            generate_dummy_dataset,
            check_new_data,
            run_autogluon_search,
            deploy_best_model,
            create_automl_dag
        )
        
        # All functions should be callable
        assert callable(generate_dummy_dataset)
        assert callable(check_new_data)
        assert callable(run_autogluon_search)
        assert callable(deploy_best_model)
        assert callable(create_automl_dag)


# Test fixtures
@pytest.fixture
def sample_data_directory():
    """Fixture to create a temporary directory with sample data"""
    tmpdir = tempfile.mkdtemp()
    
    # Create sample CSV
    sample_df = pd.DataFrame({
        'feature_1': range(10),
        'feature_2': range(10, 20),
        'target': [0, 1] * 5
    })
    csv_path = os.path.join(tmpdir, 'sample.csv')
    sample_df.to_csv(csv_path, index=False)
    
    yield tmpdir
    
    # Cleanup
    shutil.rmtree(tmpdir)


def test_with_fixture(sample_data_directory):
    """Test using the fixture"""
    result = check_new_data(data_directory=sample_data_directory)
    assert result is not None
    assert result.endswith('.csv')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

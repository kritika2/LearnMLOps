"""
Unit tests for AutoTuner class
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '03-hyperparameter-tuning'))

from auto_tuner import AutoTuner, OPTUNA_AVAILABLE


class TestAutoTuner:
    """Test suite for AutoTuner"""
    
    @pytest.fixture
    def tuner(self):
        """Create an AutoTuner instance for testing"""
        return AutoTuner()
    
    @pytest.fixture
    def classification_data(self):
        """Generate sample classification data"""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    @pytest.fixture
    def regression_data(self):
        """Generate sample regression data"""
        X, y = make_regression(
            n_samples=500,
            n_features=10,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    def test_initialization(self, tuner):
        """Test AutoTuner initialization"""
        assert tuner is not None
        assert tuner.model_type == 'auto'
        assert tuner.best_params is None
        assert tuner.best_score is None
        assert tuner.tuning_history == []
        assert tuner.optimization_time == 0
    
    def test_initialization_with_model_type(self):
        """Test AutoTuner initialization with specific model type"""
        tuner = AutoTuner(model_type='random_forest_clf')
        assert tuner.model_type == 'random_forest_clf'
    
    def test_param_spaces_exist(self, tuner):
        """Test that parameter spaces are properly defined"""
        assert 'random_forest_clf' in tuner.param_spaces
        assert 'xgboost_clf' in tuner.param_spaces
        assert 'logistic_regression' in tuner.param_spaces
        
        # Check that each model has grid, random, and bayesian spaces
        for model_type in ['random_forest_clf', 'xgboost_clf', 'logistic_regression']:
            assert 'grid' in tuner.param_spaces[model_type]
            assert 'random' in tuner.param_spaces[model_type]
            assert 'bayesian' in tuner.param_spaces[model_type]
    
    def test_detect_model_type_random_forest(self, tuner):
        """Test model type detection for Random Forest"""
        rf_clf = RandomForestClassifier()
        detected = tuner.detect_model_type(rf_clf)
        assert detected == 'random_forest_clf'
        
        rf_reg = RandomForestRegressor()
        detected = tuner.detect_model_type(rf_reg)
        assert detected == 'random_forest_reg'
    
    def test_detect_model_type_logistic_regression(self, tuner):
        """Test model type detection for Logistic Regression"""
        lr = LogisticRegression()
        detected = tuner.detect_model_type(lr)
        assert detected == 'logistic_regression'
    
    def test_detect_model_type_unknown(self, tuner):
        """Test model type detection for unknown models"""
        class CustomModel:
            pass
        
        model = CustomModel()
        detected = tuner.detect_model_type(model)
        assert detected == 'unknown'
    
    def test_grid_search(self, tuner, classification_data):
        """Test grid search tuning"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        result = tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='grid',
            cv=3,
            scoring='accuracy'
        )
        
        assert result is not None
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'best_model' in result
        assert 'cv_results' in result
        assert 'n_combinations' in result
        
        # Check that best params were set
        assert tuner.best_params is not None
        assert tuner.best_score is not None
        assert tuner.best_score > 0
    
    def test_random_search(self, tuner, classification_data):
        """Test random search tuning"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        result = tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='random',
            cv=3,
            n_trials=10,
            scoring='accuracy'
        )
        
        assert result is not None
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'best_model' in result
        assert result['n_combinations'] == 10
        
        assert tuner.best_params is not None
        assert tuner.best_score > 0
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
    def test_bayesian_optimization(self, tuner, classification_data):
        """Test Bayesian optimization tuning"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        result = tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='bayesian',
            cv=3,
            n_trials=10,
            scoring='accuracy'
        )
        
        assert result is not None
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'best_model' in result
        assert 'study' in result
        
        assert tuner.best_params is not None
        assert tuner.best_score > 0
    
    def test_bayesian_fallback_without_optuna(self, tuner, classification_data):
        """Test that Bayesian optimization falls back to random search without Optuna"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        
        # This should work regardless of whether Optuna is available
        result = tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='bayesian',
            cv=3,
            n_trials=5,
            scoring='accuracy'
        )
        
        assert result is not None
        assert 'best_params' in result
        assert 'best_score' in result
    
    def test_tune_logistic_regression(self, tuner, classification_data):
        """Test tuning Logistic Regression"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        result = tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='random',
            cv=3,
            n_trials=5,
            scoring='accuracy'
        )
        
        assert result is not None
        assert 'best_params' in result
        assert 'C' in result['best_params']
        assert 'penalty' in result['best_params']
    
    def test_tune_regression_model(self, tuner, regression_data):
        """Test tuning regression model"""
        X_train, X_test, y_train, y_test = regression_data
        
        model = RandomForestRegressor(random_state=42)
        result = tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='random',
            cv=3,
            n_trials=5,
            scoring='r2'
        )
        
        assert result is not None
        assert 'best_params' in result
        assert tuner.best_score is not None
    
    def test_optimization_time_tracking(self, tuner, classification_data):
        """Test that optimization time is tracked"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='random',
            cv=3,
            n_trials=5
        )
        
        assert tuner.optimization_time > 0
    
    def test_tuning_history(self, tuner, classification_data):
        """Test that tuning history is recorded"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='random',
            cv=3,
            n_trials=5
        )
        
        assert len(tuner.tuning_history) > 0
    
    def test_compare_strategies(self, tuner, classification_data):
        """Test strategy comparison"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        
        # Use only random search for faster testing
        comparison = tuner.compare_strategies(
            model, X_train, y_train,
            strategies=['random'],
            cv=3,
            max_time_per_strategy=30
        )
        
        assert 'random' in comparison
        assert 'best_score' in comparison['random']
        assert 'best_params' in comparison['random']
        assert 'optimization_time' in comparison['random']
    
    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
    def test_compare_multiple_strategies(self, tuner, classification_data):
        """Test comparing multiple strategies"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        
        comparison = tuner.compare_strategies(
            model, X_train, y_train,
            strategies=['random', 'bayesian'],
            cv=3,
            max_time_per_strategy=30
        )
        
        assert len(comparison) == 2
        for strategy in ['random', 'bayesian']:
            assert strategy in comparison
            if 'error' not in comparison[strategy]:
                assert 'best_score' in comparison[strategy]
    
    def test_invalid_strategy_error(self, tuner, classification_data):
        """Test error handling for invalid strategy"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            tuner.tune_hyperparameters(
                model, X_train, y_train,
                strategy='invalid_strategy'
            )
    
    def test_unsupported_model_type_error(self, tuner, classification_data):
        """Test error handling for unsupported model type"""
        X_train, X_test, y_train, y_test = classification_data
        
        class UnsupportedModel:
            def __init__(self):
                pass
        
        model = UnsupportedModel()
        tuner.model_type = 'unsupported_model'
        
        with pytest.raises(ValueError, match="not supported"):
            tuner.tune_hyperparameters(
                model, X_train, y_train,
                strategy='random'
            )
    
    def test_best_model_performance(self, tuner, classification_data):
        """Test that best model performs well on test set"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        result = tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='random',
            cv=3,
            n_trials=5,
            scoring='accuracy'
        )
        
        best_model = result['best_model']
        best_model.fit(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        
        # Test score should be reasonable
        assert test_score > 0.5
    
    def test_different_cv_folds(self, tuner, classification_data):
        """Test tuning with different CV folds"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        
        # Test with 3-fold CV
        result_3 = tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='random',
            cv=3,
            n_trials=5
        )
        assert result_3 is not None
        
        # Test with 5-fold CV
        tuner2 = AutoTuner()
        result_5 = tuner2.tune_hyperparameters(
            model, X_train, y_train,
            strategy='random',
            cv=5,
            n_trials=5
        )
        assert result_5 is not None
    
    def test_custom_scoring_metric(self, tuner, classification_data):
        """Test tuning with custom scoring metric"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        
        # Test with f1 score
        result = tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='random',
            cv=3,
            n_trials=5,
            scoring='f1_macro'
        )
        
        assert result is not None
        assert tuner.best_score is not None
    
    def test_timeout_parameter(self, tuner, classification_data):
        """Test timeout parameter functionality"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        
        # Set a very short timeout
        result = tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='random',
            cv=3,
            n_trials=100,
            timeout=5  # 5 seconds
        )
        
        # Should complete within timeout
        assert tuner.optimization_time < 10
    
    def test_parameter_space_random_forest(self, tuner):
        """Test parameter space structure for Random Forest"""
        param_space = tuner.param_spaces['random_forest_clf']
        
        # Check grid space
        assert 'n_estimators' in param_space['grid']
        assert 'max_depth' in param_space['grid']
        assert 'min_samples_split' in param_space['grid']
        
        # Check random space
        assert 'n_estimators' in param_space['random']
        assert 'max_depth' in param_space['random']
        
        # Check bayesian space
        assert 'n_estimators' in param_space['bayesian']
        assert param_space['bayesian']['n_estimators']['type'] == 'int'
    
    def test_parameter_space_logistic_regression(self, tuner):
        """Test parameter space structure for Logistic Regression"""
        param_space = tuner.param_spaces['logistic_regression']
        
        # Check grid space
        assert 'C' in param_space['grid']
        assert 'penalty' in param_space['grid']
        assert 'solver' in param_space['grid']
        
        # Check bayesian space
        assert 'C' in param_space['bayesian']
        assert param_space['bayesian']['C']['type'] == 'float'
    
    def test_small_dataset(self, tuner):
        """Test tuning on small dataset"""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        model = RandomForestClassifier(random_state=42)
        result = tuner.tune_hyperparameters(
            model, X, y,
            strategy='random',
            cv=3,
            n_trials=5
        )
        
        assert result is not None
        assert tuner.best_score is not None
    
    def test_multiclass_classification(self, tuner):
        """Test tuning for multiclass classification"""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_classes=5,
            n_informative=8,
            random_state=42
        )
        
        model = RandomForestClassifier(random_state=42)
        result = tuner.tune_hyperparameters(
            model, X, y,
            strategy='random',
            cv=3,
            n_trials=5,
            scoring='accuracy'
        )
        
        assert result is not None
        assert tuner.best_score > 0
    
    def test_best_params_format(self, tuner, classification_data):
        """Test that best parameters have correct format"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        result = tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='random',
            cv=3,
            n_trials=5
        )
        
        best_params = result['best_params']
        
        # Check parameter types
        assert isinstance(best_params['n_estimators'], int)
        assert best_params['n_estimators'] > 0
        
        if 'max_depth' in best_params and best_params['max_depth'] is not None:
            assert isinstance(best_params['max_depth'], int)
            assert best_params['max_depth'] > 0
    
    def test_cv_results_dataframe(self, tuner, classification_data):
        """Test that CV results are returned as DataFrame"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(random_state=42)
        result = tuner.tune_hyperparameters(
            model, X_train, y_train,
            strategy='grid',
            cv=3
        )
        
        assert 'cv_results' in result
        assert isinstance(result['cv_results'], pd.DataFrame)
        assert len(result['cv_results']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


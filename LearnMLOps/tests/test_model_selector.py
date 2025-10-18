"""
Unit tests for ModelSelector class
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01-model-selection'))

from model_selector import ModelSelector


class TestModelSelector:
    """Test suite for ModelSelector"""
    
    @pytest.fixture
    def selector(self):
        """Create a ModelSelector instance for testing"""
        return ModelSelector()
    
    @pytest.fixture
    def classification_data(self):
        """Generate sample classification data"""
        X, y = make_classification(
            n_samples=1000, 
            n_features=10, 
            n_classes=3, 
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def regression_data(self):
        """Generate sample regression data"""
        X, y = make_regression(
            n_samples=500, 
            n_features=5, 
            noise=0.1, 
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def clustering_data(self):
        """Generate sample clustering data"""
        X, _ = make_blobs(
            n_samples=300, 
            centers=4, 
            n_features=2, 
            random_state=42
        )
        return X
    
    def test_initialization(self, selector):
        """Test ModelSelector initialization"""
        assert selector is not None
        assert 'regression' in selector.supervised_models
        assert 'classification' in selector.supervised_models
        assert 'clustering' in selector.unsupervised_models
        assert 'anomaly_detection' in selector.unsupervised_models
    
    def test_analyze_data_classification(self, selector, classification_data):
        """Test data analysis for classification problem"""
        X, y = classification_data
        analysis = selector.analyze_data(X, y)
        
        assert analysis['n_samples'] == 1000
        assert analysis['n_features'] == 10
        assert analysis['has_labels'] is True
        assert analysis['problem_type'] == 'classification'
        assert analysis['target_type'] == 'categorical'
        assert analysis['n_classes'] == 3
    
    def test_analyze_data_regression(self, selector, regression_data):
        """Test data analysis for regression problem"""
        X, y = regression_data
        analysis = selector.analyze_data(X, y)
        
        assert analysis['n_samples'] == 500
        assert analysis['n_features'] == 5
        assert analysis['has_labels'] is True
        assert analysis['problem_type'] == 'regression'
        assert analysis['target_type'] == 'continuous'
    
    def test_analyze_data_unsupervised(self, selector, clustering_data):
        """Test data analysis for unsupervised problem"""
        X = clustering_data
        analysis = selector.analyze_data(X)
        
        assert analysis['n_samples'] == 300
        assert analysis['n_features'] == 2
        assert analysis['has_labels'] is False
        assert analysis['problem_type'] == 'unsupervised'
    
    def test_categorize_size_small(self, selector):
        """Test dataset size categorization - small"""
        assert selector._categorize_size(500) == 'small'
        assert selector._categorize_size(999) == 'small'
    
    def test_categorize_size_medium(self, selector):
        """Test dataset size categorization - medium"""
        assert selector._categorize_size(1000) == 'medium'
        assert selector._categorize_size(50000) == 'medium'
        assert selector._categorize_size(99999) == 'medium'
    
    def test_categorize_size_large(self, selector):
        """Test dataset size categorization - large"""
        assert selector._categorize_size(100000) == 'large'
        assert selector._categorize_size(1000000) == 'large'
    
    def test_is_continuous_true(self, selector):
        """Test continuous variable detection - continuous"""
        y_continuous = np.random.randn(1000)
        assert selector._is_continuous(y_continuous) is True
    
    def test_is_continuous_false_categorical(self, selector):
        """Test continuous variable detection - categorical"""
        y_categorical = np.array([0, 1, 2] * 100)
        assert selector._is_continuous(y_categorical) is False
    
    def test_is_continuous_false_binary(self, selector):
        """Test continuous variable detection - binary"""
        y_binary = np.array([0, 1] * 50)
        assert selector._is_continuous(y_binary) is False
    
    def test_suggest_models_classification(self, selector, classification_data):
        """Test model suggestions for classification"""
        X, y = classification_data
        suggestions, analysis = selector.suggest_models(X, y, top_k=3)
        
        assert len(suggestions) <= 3
        assert analysis['problem_type'] == 'classification'
        assert all(isinstance(s, str) for s in suggestions)
    
    def test_suggest_models_regression(self, selector, regression_data):
        """Test model suggestions for regression"""
        X, y = regression_data
        suggestions, analysis = selector.suggest_models(X, y, top_k=3)
        
        assert len(suggestions) <= 3
        assert analysis['problem_type'] == 'regression'
        assert all(isinstance(s, str) for s in suggestions)
    
    def test_suggest_models_unsupervised(self, selector, clustering_data):
        """Test model suggestions for unsupervised learning"""
        X = clustering_data
        suggestions, analysis = selector.suggest_models(X, top_k=3)
        
        assert len(suggestions) <= 3
        assert analysis['problem_type'] == 'unsupervised'
        assert all(isinstance(s, str) for s in suggestions)
    
    def test_suggest_models_small_dataset_regression(self, selector):
        """Test suggestions for small regression dataset"""
        X, y = make_regression(n_samples=500, n_features=5, random_state=42)
        suggestions, _ = selector.suggest_models(X, y)
        
        # Small regression should suggest Linear Regression or SVR
        assert any(s in ['Linear Regression', 'SVR'] for s in suggestions)
    
    def test_suggest_models_small_dataset_classification(self, selector):
        """Test suggestions for small classification dataset"""
        X, y = make_classification(n_samples=800, n_features=10, random_state=42)
        suggestions, _ = selector.suggest_models(X, y)
        
        # Small classification should suggest Logistic Regression or Naive Bayes
        assert any(s in ['Logistic Regression', 'Naive Bayes'] for s in suggestions)
    
    def test_suggest_models_large_dataset(self, selector):
        """Test suggestions for large dataset"""
        X, y = make_classification(n_samples=150000, n_features=10, random_state=42)
        suggestions, _ = selector.suggest_models(X, y, top_k=2)
        
        # Large dataset should suggest Random Forest
        assert 'Random Forest' in suggestions
    
    def test_evaluate_models_classification(self, selector, classification_data):
        """Test model evaluation for classification"""
        X, y = classification_data
        results = selector.evaluate_models(X, y, cv=3)
        
        assert results is not None
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that results contain expected keys
        for model_name, metrics in results.items():
            assert 'mean_score' in metrics
            assert 'std_score' in metrics
            assert isinstance(metrics['mean_score'], (int, float))
    
    def test_evaluate_models_regression(self, selector, regression_data):
        """Test model evaluation for regression"""
        X, y = regression_data
        results = selector.evaluate_models(X, y, cv=3)
        
        assert results is not None
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Verify scores are in reasonable range for R2
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                # R2 score should be between -inf and 1, but typically > -1
                assert metrics['mean_score'] > -10
    
    def test_evaluate_models_unsupervised_returns_none(self, selector, clustering_data):
        """Test that unsupervised models cannot be evaluated with cross-validation"""
        X = clustering_data
        results = selector.evaluate_models(X, y=None, cv=3)
        
        assert results is None
    
    def test_evaluate_models_different_cv(self, selector, classification_data):
        """Test model evaluation with different CV folds"""
        X, y = classification_data
        
        # Test with 3-fold CV
        results_3 = selector.evaluate_models(X, y, cv=3)
        assert results_3 is not None
        
        # Test with 5-fold CV
        results_5 = selector.evaluate_models(X, y, cv=5)
        assert results_5 is not None
        
        # Results should have same models
        assert set(results_3.keys()) == set(results_5.keys())
    
    def test_empty_dataset_error(self, selector):
        """Test behavior with empty dataset"""
        X = np.array([]).reshape(0, 10)
        y = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            selector.analyze_data(X, y)
    
    def test_mismatched_dimensions(self, selector):
        """Test behavior with mismatched X and y dimensions"""
        X = np.random.randn(100, 10)
        y = np.random.randn(50)  # Wrong size
        
        with pytest.raises((ValueError, IndexError)):
            selector.evaluate_models(X, y)
    
    def test_single_feature(self, selector):
        """Test with single feature dataset"""
        X = np.random.randn(100, 1)
        y = np.random.choice([0, 1], 100)
        
        analysis = selector.analyze_data(X, y)
        assert analysis['n_features'] == 1
        
        suggestions, _ = selector.suggest_models(X, y)
        assert len(suggestions) > 0
    
    def test_binary_classification(self, selector):
        """Test binary classification detection"""
        X, y = make_classification(
            n_samples=1000, 
            n_features=10, 
            n_classes=2,  # Binary
            random_state=42
        )
        
        analysis = selector.analyze_data(X, y)
        assert analysis['problem_type'] == 'classification'
        assert analysis['n_classes'] == 2
    
    def test_multiclass_classification(self, selector):
        """Test multiclass classification detection"""
        X, y = make_classification(
            n_samples=1000, 
            n_features=10, 
            n_classes=5,  # Multi-class
            random_state=42
        )
        
        analysis = selector.analyze_data(X, y)
        assert analysis['problem_type'] == 'classification'
        assert analysis['n_classes'] == 5
    
    def test_high_dimensional_data(self, selector):
        """Test with high-dimensional data (more features than samples)"""
        X = np.random.randn(50, 100)  # 50 samples, 100 features
        
        analysis = selector.analyze_data(X)
        assert analysis['feature_density'] == 'high'
        assert analysis['n_features'] > analysis['n_samples']
    
    def test_supervised_model_pools_exist(self, selector):
        """Test that all supervised model pools are properly initialized"""
        assert 'Linear Regression' in selector.supervised_models['regression']
        assert 'Random Forest' in selector.supervised_models['regression']
        assert 'SVR' in selector.supervised_models['regression']
        
        assert 'Logistic Regression' in selector.supervised_models['classification']
        assert 'Random Forest' in selector.supervised_models['classification']
        assert 'SVM' in selector.supervised_models['classification']
        assert 'Naive Bayes' in selector.supervised_models['classification']
    
    def test_unsupervised_model_pools_exist(self, selector):
        """Test that all unsupervised model pools are properly initialized"""
        assert 'K-Means' in selector.unsupervised_models['clustering']
        assert 'DBSCAN' in selector.unsupervised_models['clustering']
        
        assert 'Isolation Forest' in selector.unsupervised_models['anomaly_detection']
        assert 'Local Outlier Factor' in selector.unsupervised_models['anomaly_detection']
        assert 'One-Class SVM' in selector.unsupervised_models['anomaly_detection']
        
        assert 'PCA' in selector.unsupervised_models['dimensionality_reduction']
    
    def test_anomaly_detection_suggestion(self, selector):
        """Test anomaly detection model suggestions"""
        # Create large dataset that should trigger anomaly detection suggestions
        X = np.random.randn(12000, 10)
        
        suggestions, analysis = selector.suggest_models(X)
        
        # Should suggest anomaly detection or dimensionality reduction models
        assert any(s in ['Isolation Forest', 'PCA', 'K-Means'] for s in suggestions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


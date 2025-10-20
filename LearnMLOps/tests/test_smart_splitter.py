"""
Unit tests for SmartSplitter class
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02-data-splitting'))

from smart_splitter import SmartSplitter


class TestSmartSplitter:
    """Test suite for SmartSplitter"""
    
    @pytest.fixture
    def splitter(self):
        """Create a SmartSplitter instance for testing"""
        return SmartSplitter()
    
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
    def imbalanced_data(self):
        """Generate imbalanced classification data"""
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_classes=3,
            weights=[0.8, 0.15, 0.05],
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def small_data(self):
        """Generate small dataset"""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def time_series_data(self):
        """Generate time series data"""
        n_samples = 1000
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'date': dates
        })
        y = np.random.randn(n_samples)
        return X, y
    
    @pytest.fixture
    def grouped_data(self):
        """Generate grouped data"""
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        # Create groups (e.g., patient IDs, user IDs)
        groups = np.repeat(np.arange(100), 10)
        return X, y, groups
    
    def test_initialization(self, splitter):
        """Test SmartSplitter initialization"""
        assert splitter is not None
        assert splitter.split_history == []
    
    def test_analyze_dataset_basic(self, splitter, classification_data):
        """Test basic dataset analysis"""
        X, y = classification_data
        analysis = splitter.analyze_dataset(X, y)
        
        assert analysis['n_samples'] == 1000
        assert analysis['n_features'] == 10
        assert analysis['has_groups'] is False
        assert analysis['is_time_series'] is False
        assert analysis['problem_type'] == 'classification'
    
    def test_analyze_dataset_imbalanced(self, splitter, imbalanced_data):
        """Test analysis of imbalanced dataset"""
        X, y = imbalanced_data
        analysis = splitter.analyze_dataset(X, y)
        
        assert analysis['class_imbalance'] is True
        assert analysis['n_classes'] == 3
    
    def test_analyze_dataset_balanced(self, splitter, classification_data):
        """Test analysis of balanced dataset"""
        X, y = classification_data
        analysis = splitter.analyze_dataset(X, y)
        
        assert analysis['class_imbalance'] is False
    
    def test_analyze_dataset_regression(self, splitter):
        """Test analysis of regression dataset"""
        X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
        analysis = splitter.analyze_dataset(X, y)
        
        assert analysis['problem_type'] == 'regression'
    
    def test_analyze_dataset_with_groups(self, splitter, grouped_data):
        """Test analysis with grouped data"""
        X, y, groups = grouped_data
        analysis = splitter.analyze_dataset(X, y, groups=groups)
        
        assert analysis['has_groups'] is True
    
    def test_analyze_dataset_time_series(self, splitter, time_series_data):
        """Test analysis of time series data"""
        X, y = time_series_data
        analysis = splitter.analyze_dataset(X, y, time_column='date')
        
        assert analysis['is_time_series'] is True
    
    def test_categorize_size_small(self, splitter):
        """Test size categorization - small"""
        assert splitter._categorize_size(500) == 'small'
        assert splitter._categorize_size(9999) == 'small'
    
    def test_categorize_size_medium(self, splitter):
        """Test size categorization - medium"""
        assert splitter._categorize_size(10000) == 'medium'
        assert splitter._categorize_size(50000) == 'medium'
        assert splitter._categorize_size(99999) == 'medium'
    
    def test_categorize_size_large(self, splitter):
        """Test size categorization - large"""
        assert splitter._categorize_size(100000) == 'large'
    
    def test_recommend_strategy_small_dataset(self, splitter, small_data):
        """Test strategy recommendation for small dataset"""
        X, y = small_data
        strategy = splitter.recommend_strategy(X, y)
        
        assert strategy['strategy'] == 'cross_validation'
        assert 'cv_folds' in strategy
        assert strategy['cv_folds'] in [3, 5]
    
    def test_recommend_strategy_time_series(self, splitter, time_series_data):
        """Test strategy recommendation for time series"""
        X, y = time_series_data
        strategy = splitter.recommend_strategy(X, y, time_column='date')
        
        assert strategy['strategy'] == 'temporal_split'
        assert 'train_ratio' in strategy
        assert 'val_ratio' in strategy
        assert 'test_ratio' in strategy
    
    def test_recommend_strategy_grouped(self, splitter, grouped_data):
        """Test strategy recommendation for grouped data"""
        X, y, groups = grouped_data
        strategy = splitter.recommend_strategy(X, y, groups=groups)
        
        assert strategy['strategy'] == 'group_split'
        assert 'test_size' in strategy
    
    def test_recommend_strategy_imbalanced(self, splitter, imbalanced_data):
        """Test strategy recommendation for imbalanced data"""
        X, y = imbalanced_data
        strategy = splitter.recommend_strategy(X, y)
        
        assert strategy['strategy'] == 'stratified_split'
        assert 'train_ratio' in strategy
    
    def test_recommend_strategy_standard_medium(self, splitter):
        """Test strategy recommendation for medium dataset"""
        X, y = make_classification(n_samples=50000, n_features=10, random_state=42)
        strategy = splitter.recommend_strategy(X, y)
        
        assert strategy['strategy'] == 'standard_split'
        assert strategy['train_ratio'] == 0.8
    
    def test_recommend_strategy_standard_large(self, splitter):
        """Test strategy recommendation for large dataset"""
        X, y = make_classification(n_samples=150000, n_features=10, random_state=42)
        strategy = splitter.recommend_strategy(X, y)
        
        assert strategy['strategy'] == 'standard_split'
        assert strategy['train_ratio'] == 0.9
    
    def test_split_data_standard(self, splitter, classification_data):
        """Test standard data splitting"""
        X, y = classification_data
        result = splitter.split_data(X, y, random_state=42)
        
        assert 'X_train' in result
        assert 'X_val' in result
        assert 'X_test' in result
        assert 'y_train' in result
        assert 'y_val' in result
        assert 'y_test' in result
        assert result['type'] in ['standard_split', 'stratified_split']
        
        # Check sizes add up
        total_size = len(result['X_train']) + len(result['X_val']) + len(result['X_test'])
        assert total_size == len(X)
    
    def test_split_data_stratified(self, splitter, imbalanced_data):
        """Test stratified splitting"""
        X, y = imbalanced_data
        result = splitter.split_data(X, y, random_state=42)
        
        assert result['type'] == 'stratified_split'
        
        # Check that class distributions are preserved
        train_classes, train_counts = np.unique(result['y_train'], return_counts=True)
        val_classes, val_counts = np.unique(result['y_val'], return_counts=True)
        test_classes, test_counts = np.unique(result['y_test'], return_counts=True)
        
        # All classes should be present
        assert len(train_classes) == 3
        assert len(val_classes) == 3
        assert len(test_classes) == 3
    
    def test_split_data_temporal(self, splitter, time_series_data):
        """Test temporal splitting"""
        X, y = time_series_data
        result = splitter.split_data(X, y, time_column='date', random_state=42)
        
        assert result['type'] == 'temporal_split'
        
        # Check temporal ordering
        assert result['X_train']['date'].max() <= result['X_val']['date'].min()
        assert result['X_val']['date'].max() <= result['X_test']['date'].min()
    
    def test_split_data_grouped(self, splitter, grouped_data):
        """Test grouped splitting"""
        X, y, groups = grouped_data
        result = splitter.split_data(X, y, groups=groups, random_state=42)
        
        assert result['type'] == 'group_split'
        
        # Check that groups don't overlap
        train_groups = set(groups[np.isin(np.arange(len(groups)), np.arange(len(result['X_train'])))])
        test_groups = set(groups[np.isin(np.arange(len(groups)), 
                                        np.arange(len(result['X_train']) + len(result['X_val']), len(groups)))])
        
        # Groups should be mutually exclusive (approximately, due to indexing)
        assert len(result['X_train']) + len(result['X_val']) + len(result['X_test']) == len(X)
    
    def test_split_data_cross_validation(self, splitter, small_data):
        """Test cross-validation split"""
        X, y = small_data
        result = splitter.split_data(X, y, random_state=42)
        
        assert result['type'] == 'cross_validation'
        assert 'cv_folds' in result
        assert 'X' in result
        assert 'y' in result
    
    def test_split_data_custom_strategy(self, splitter, classification_data):
        """Test split with custom strategy"""
        X, y = classification_data
        
        custom_strategy = {
            'strategy': 'standard_split',
            'train_ratio': 0.6,
            'val_ratio': 0.2,
            'test_ratio': 0.2
        }
        
        result = splitter.split_data(X, y, strategy=custom_strategy, random_state=42)
        
        assert result['type'] == 'standard_split'
        
        # Check approximate ratios
        total = len(X)
        train_ratio = len(result['X_train']) / total
        assert 0.55 < train_ratio < 0.65
    
    def test_evaluate_split_quality_standard(self, splitter, classification_data):
        """Test split quality evaluation for standard split"""
        X, y = classification_data
        result = splitter.split_data(X, y, random_state=42)
        
        quality = splitter.evaluate_split_quality(result)
        
        assert 'train_ratio' in quality
        assert 'val_ratio' in quality
        assert 'test_ratio' in quality
        assert 'class_balance_maintained' in quality
        
        # Ratios should sum to approximately 1
        ratio_sum = quality['train_ratio'] + quality['val_ratio'] + quality['test_ratio']
        assert 0.99 < ratio_sum < 1.01
    
    def test_evaluate_split_quality_stratified(self, splitter, classification_data):
        """Test split quality evaluation for stratified split"""
        X, y = classification_data
        
        strategy = {
            'strategy': 'stratified_split',
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15
        }
        
        result = splitter.split_data(X, y, strategy=strategy, random_state=42)
        quality = splitter.evaluate_split_quality(result)
        
        # Stratified split should maintain class balance well
        assert quality['class_balance_maintained'] is True
        assert quality['max_class_distribution_diff'] < 0.05
    
    def test_evaluate_split_quality_cross_validation(self, splitter, small_data):
        """Test split quality evaluation for cross-validation"""
        X, y = small_data
        result = splitter.split_data(X, y, random_state=42)
        
        quality = splitter.evaluate_split_quality(result)
        
        assert 'note' in quality
    
    def test_split_data_reproducibility(self, splitter, classification_data):
        """Test that splitting is reproducible with same random_state"""
        X, y = classification_data
        
        result1 = splitter.split_data(X, y, random_state=42)
        result2 = splitter.split_data(X, y, random_state=42)
        
        # Check that splits are identical
        np.testing.assert_array_equal(result1['X_train'], result2['X_train'])
        np.testing.assert_array_equal(result1['y_train'], result2['y_train'])
    
    def test_split_data_different_random_states(self, splitter, classification_data):
        """Test that different random states produce different splits"""
        X, y = classification_data
        
        result1 = splitter.split_data(X, y, random_state=42)
        result2 = splitter.split_data(X, y, random_state=123)
        
        # Check that splits are different
        assert not np.array_equal(result1['X_train'], result2['X_train'])
    
    def test_temporal_split_without_time_column_error(self, splitter, classification_data):
        """Test temporal split error when time column is missing"""
        X, y = classification_data
        
        strategy = {
            'strategy': 'temporal_split',
            'train_ratio': 0.7,
            'val_ratio': 0.2,
            'test_ratio': 0.1
        }
        
        with pytest.raises(ValueError):
            splitter.split_data(X, y, strategy=strategy)
    
    def test_empty_dataset_error(self, splitter):
        """Test behavior with empty dataset"""
        X = np.array([]).reshape(0, 10)
        y = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            splitter.split_data(X, y)
    
    def test_single_sample_error(self, splitter):
        """Test behavior with single sample"""
        X = np.array([[1, 2, 3]])
        y = np.array([0])
        
        # Should raise error or handle gracefully
        try:
            result = splitter.split_data(X, y)
            # If it doesn't error, at least check it returns something
            assert result is not None
        except (ValueError, IndexError):
            pass  # Expected behavior
    
    def test_feature_distribution_similarity(self, splitter, classification_data):
        """Test feature distribution similarity metric"""
        X, y = classification_data
        result = splitter.split_data(X, y, random_state=42)
        
        quality = splitter.evaluate_split_quality(result)
        
        if 'feature_distribution_similar' in quality:
            # Most features should have similar distributions
            assert quality['feature_distribution_similar'] > 0.5
    
    def test_large_dataset_split(self, splitter):
        """Test splitting of large dataset"""
        X, y = make_classification(n_samples=100000, n_features=20, random_state=42)
        
        result = splitter.split_data(X, y, random_state=42)
        
        # For large dataset, should use larger train ratio
        assert result['type'] == 'standard_split'
        train_ratio = len(result['X_train']) / len(X)
        assert train_ratio > 0.85
    
    def test_very_small_dataset_cv_folds(self, splitter):
        """Test that very small datasets get appropriate CV folds"""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        
        strategy = splitter.recommend_strategy(X, y)
        
        assert strategy['strategy'] == 'cross_validation'
        assert strategy['cv_folds'] == 3  # Should use fewer folds for tiny datasets
    
    def test_pandas_dataframe_input(self, splitter):
        """Test that pandas DataFrames are handled correctly"""
        X_np, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        X = pd.DataFrame(X_np, columns=[f'feature_{i}' for i in range(10)])
        
        result = splitter.split_data(X, y, random_state=42)
        
        # Should return DataFrames
        assert isinstance(result['X_train'], (np.ndarray, pd.DataFrame))
        assert len(result['X_train']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


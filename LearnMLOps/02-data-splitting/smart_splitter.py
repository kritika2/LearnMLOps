"""
Smart Data Splitter
Intelligent data splitting based on dataset characteristics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, 
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    GroupShuffleSplit,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

class SmartSplitter:
    """
    Intelligent data splitting based on dataset and problem characteristics
    """
    
    def __init__(self):
        self.split_history = []
    
    def analyze_dataset(self, X, y=None, groups=None, time_column=None):
        """
        Analyze dataset to determine optimal splitting strategy
        """
        n_samples, n_features = X.shape
        
        analysis = {
            'n_samples': n_samples,
            'n_features': n_features,
            'dataset_size_category': self._categorize_size(n_samples),
            'has_groups': groups is not None,
            'is_time_series': time_column is not None,
            'class_imbalance': None
        }
        
        if y is not None:
            unique_classes = np.unique(y)
            if len(unique_classes) <= 20:  # Likely classification
                class_counts = pd.Series(y).value_counts()
                imbalance_ratio = class_counts.min() / class_counts.max()
                analysis['class_imbalance'] = imbalance_ratio < 0.1
                analysis['n_classes'] = len(unique_classes)
                analysis['problem_type'] = 'classification'
            else:
                analysis['problem_type'] = 'regression'
        
        return analysis
    
    def recommend_strategy(self, X, y=None, groups=None, time_column=None):
        """
        Recommend optimal splitting strategy based on data analysis
        """
        analysis = self.analyze_dataset(X, y, groups, time_column)
        
        # Small dataset - use cross-validation
        if analysis['n_samples'] < 1000:
            return {
                'strategy': 'cross_validation',
                'cv_folds': 5 if analysis['n_samples'] > 100 else 3,
                'rationale': 'Dataset too small for reliable train/test split'
            }
        
        # Time series data
        if analysis['is_time_series']:
            return {
                'strategy': 'temporal_split',
                'train_ratio': 0.7,
                'val_ratio': 0.2,
                'test_ratio': 0.1,
                'rationale': 'Time series requires temporal ordering'
            }
        
        # Grouped data (e.g., multiple samples per patient/user)
        if analysis['has_groups']:
            return {
                'strategy': 'group_split',
                'test_size': 0.2,
                'rationale': 'Grouped data requires group-level splitting'
            }
        
        # Imbalanced classification
        if analysis.get('class_imbalance', False):
            return {
                'strategy': 'stratified_split',
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'rationale': 'Imbalanced classes require stratified splitting'
            }
        
        # Standard splitting based on size
        size_category = analysis['dataset_size_category']
        
        if size_category == 'small':
            return {
                'strategy': 'standard_split',
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'rationale': 'Standard split for small-medium dataset'
            }
        elif size_category == 'medium':
            return {
                'strategy': 'standard_split',
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1,
                'rationale': 'More training data for medium dataset'
            }
        else:  # large
            return {
                'strategy': 'standard_split',
                'train_ratio': 0.9,
                'val_ratio': 0.08,
                'test_ratio': 0.02,
                'rationale': 'Maximum training data for large dataset'
            }
    
    def split_data(self, X, y=None, groups=None, time_column=None, 
                   strategy=None, random_state=42):
        """
        Perform intelligent data splitting
        """
        if strategy is None:
            strategy = self.recommend_strategy(X, y, groups, time_column)
        
        if strategy['strategy'] == 'cross_validation':
            return self._cross_validation_split(X, y, strategy['cv_folds'])
        
        elif strategy['strategy'] == 'temporal_split':
            return self._temporal_split(X, y, time_column, strategy)
        
        elif strategy['strategy'] == 'group_split':
            return self._group_split(X, y, groups, strategy['test_size'], random_state)
        
        elif strategy['strategy'] == 'stratified_split':
            return self._stratified_split(X, y, strategy, random_state)
        
        else:  # standard_split
            return self._standard_split(X, y, strategy, random_state)
    
    def _cross_validation_split(self, X, y, cv_folds):
        """Cross-validation for small datasets"""
        return {
            'type': 'cross_validation',
            'cv_folds': cv_folds,
            'X': X,
            'y': y,
            'note': 'Use cross_val_score for model evaluation'
        }
    
    def _temporal_split(self, X, y, time_column, strategy):
        """Temporal split preserving time order"""
        if isinstance(X, pd.DataFrame) and time_column in X.columns:
            # Sort by time
            sorted_idx = X[time_column].argsort()
            X_sorted = X.iloc[sorted_idx]
            y_sorted = y[sorted_idx] if y is not None else None
            
            n = len(X_sorted)
            train_end = int(n * strategy['train_ratio'])
            val_end = int(n * (strategy['train_ratio'] + strategy['val_ratio']))
            
            return {
                'X_train': X_sorted.iloc[:train_end],
                'X_val': X_sorted.iloc[train_end:val_end],
                'X_test': X_sorted.iloc[val_end:],
                'y_train': y_sorted[:train_end] if y is not None else None,
                'y_val': y_sorted[train_end:val_end] if y is not None else None,
                'y_test': y_sorted[val_end:] if y is not None else None,
                'type': 'temporal_split'
            }
        else:
            raise ValueError("time_column must be a column in DataFrame X")
    
    def _group_split(self, X, y, groups, test_size, random_state):
        """Group-based splitting"""
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        # Further split training into train/val
        X_temp, y_temp = X[train_idx], y[train_idx] if y is not None else None
        groups_temp = groups[train_idx]
        
        val_size = test_size / (1 - test_size)  # Adjust for remaining data
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        train_idx2, val_idx2 = next(gss_val.split(X_temp, y_temp, groups_temp))
        
        return {
            'X_train': X_temp[train_idx2],
            'X_val': X_temp[val_idx2],
            'X_test': X[test_idx],
            'y_train': y_temp[train_idx2] if y is not None else None,
            'y_val': y_temp[val_idx2] if y is not None else None,
            'y_test': y[test_idx] if y is not None else None,
            'type': 'group_split'
        }
    
    def _stratified_split(self, X, y, strategy, random_state):
        """Stratified splitting for imbalanced data"""
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(strategy['val_ratio'] + strategy['test_ratio']),
            stratify=y,
            random_state=random_state
        )
        
        # Second split: val vs test
        val_ratio_adjusted = strategy['val_ratio'] / (strategy['val_ratio'] + strategy['test_ratio'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio_adjusted),
            stratify=y_temp,
            random_state=random_state
        )
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'type': 'stratified_split'
        }
    
    def _standard_split(self, X, y, strategy, random_state):
        """Standard random splitting"""
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(strategy['val_ratio'] + strategy['test_ratio']),
            random_state=random_state
        )
        
        # Second split: val vs test
        val_ratio_adjusted = strategy['val_ratio'] / (strategy['val_ratio'] + strategy['test_ratio'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio_adjusted),
            random_state=random_state
        )
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'type': 'standard_split'
        }
    
    def evaluate_split_quality(self, split_result):
        """
        Evaluate the quality of the data split
        """
        if split_result['type'] == 'cross_validation':
            return {'note': 'Cross-validation - no split to evaluate'}
        
        X_train = split_result['X_train']
        X_val = split_result['X_val']
        X_test = split_result['X_test']
        y_train = split_result.get('y_train')
        y_val = split_result.get('y_val')
        y_test = split_result.get('y_test')
        
        metrics = {}
        
        # Size distribution
        total_size = len(X_train) + len(X_val) + len(X_test)
        metrics['train_ratio'] = len(X_train) / total_size
        metrics['val_ratio'] = len(X_val) / total_size
        metrics['test_ratio'] = len(X_test) / total_size
        
        # Class distribution balance (for classification)
        if y_train is not None and len(np.unique(y_train)) <= 20:
            train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
            val_dist = pd.Series(y_val).value_counts(normalize=True).sort_index()
            test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
            
            # Check if distributions are similar (max difference < 5%)
            train_val_diff = abs(train_dist - val_dist).max()
            train_test_diff = abs(train_dist - test_dist).max()
            
            metrics['class_balance_maintained'] = (
                train_val_diff < 0.05 and train_test_diff < 0.05
            )
            metrics['max_class_distribution_diff'] = max(train_val_diff, train_test_diff)
        
        # Feature distribution similarity (using KS test)
        if hasattr(X_train, 'shape') and X_train.shape[1] > 0:
            feature_similarities = []
            
            # Convert to numpy if needed
            if hasattr(X_train, 'values'):
                X_train_np = X_train.values
                X_test_np = X_test.values
            else:
                X_train_np = X_train
                X_test_np = X_test
            
            for i in range(min(X_train_np.shape[1], 10)):  # Check first 10 features
                try:
                    _, p_val = ks_2samp(X_train_np[:, i], X_test_np[:, i])
                    feature_similarities.append(p_val > 0.05)
                except:
                    continue
            
            if feature_similarities:
                metrics['feature_distribution_similar'] = np.mean(feature_similarities)
        
        return metrics
    
    def _categorize_size(self, n_samples):
        """Categorize dataset size"""
        if n_samples < 10000:
            return 'small'
        elif n_samples < 100000:
            return 'medium'
        else:
            return 'large'

def demo_smart_splitting():
    """
    Demonstrate smart splitting on different types of datasets
    """
    from sklearn.datasets import make_classification, make_regression
    
    print("🎯 Smart Data Splitting Demo")
    print("=" * 50)
    
    splitter = SmartSplitter()
    
    # Example 1: Small imbalanced classification dataset
    print("\n📊 Small Imbalanced Classification Dataset:")
    X_small, y_small = make_classification(
        n_samples=800, n_classes=3, weights=[0.7, 0.2, 0.1], 
        n_features=10, random_state=42
    )
    
    strategy = splitter.recommend_strategy(X_small, y_small)
    print(f"Recommended strategy: {strategy['strategy']}")
    print(f"Rationale: {strategy['rationale']}")
    
    if strategy['strategy'] != 'cross_validation':
        split_result = splitter.split_data(X_small, y_small, strategy=strategy)
        quality = splitter.evaluate_split_quality(split_result)
        print(f"Split quality: {quality}")
    
    # Example 2: Large dataset
    print("\n📈 Large Dataset:")
    X_large, y_large = make_regression(n_samples=50000, n_features=20, random_state=42)
    
    strategy = splitter.recommend_strategy(X_large, y_large)
    print(f"Recommended strategy: {strategy['strategy']}")
    print(f"Rationale: {strategy['rationale']}")
    print(f"Split ratios: Train={strategy['train_ratio']}, Val={strategy['val_ratio']}, Test={strategy['test_ratio']}")
    
    # Example 3: Time series simulation
    print("\n⏰ Time Series Dataset:")
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    X_ts = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'date': dates
    })
    y_ts = np.random.randn(1000)
    
    strategy = splitter.recommend_strategy(X_ts, y_ts, time_column='date')
    print(f"Recommended strategy: {strategy['strategy']}")
    print(f"Rationale: {strategy['rationale']}")

if __name__ == "__main__":
    demo_smart_splitting()







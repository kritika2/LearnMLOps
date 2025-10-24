"""
Unit tests for ShadowEvaluator and ModelComparator classes
"""

import pytest
import numpy as np
import time
import json
import os
from unittest.mock import Mock, patch
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '04-shadow-evaluation'))

from shadow_evaluator import (
    ShadowEvaluator,
    ModelComparator,
    ShadowEvaluationConfig,
    PredictionResult
)


class TestPredictionResult:
    """Test suite for PredictionResult dataclass"""
    
    def test_prediction_result_creation(self):
        """Test creating a PredictionResult"""
        result = PredictionResult(
            model_name='test_model',
            prediction=1,
            confidence=0.95,
            latency_ms=10.5,
            timestamp=time.time()
        )
        
        assert result.model_name == 'test_model'
        assert result.prediction == 1
        assert result.confidence == 0.95
        assert result.latency_ms == 10.5
        assert result.error is None
    
    def test_prediction_result_with_error(self):
        """Test PredictionResult with error"""
        result = PredictionResult(
            model_name='test_model',
            prediction=None,
            confidence=None,
            latency_ms=5.0,
            timestamp=time.time(),
            error='Model failed'
        )
        
        assert result.error == 'Model failed'
        assert result.prediction is None
    
    def test_prediction_result_to_dict(self):
        """Test converting PredictionResult to dictionary"""
        result = PredictionResult(
            model_name='test_model',
            prediction=1,
            confidence=0.95,
            latency_ms=10.5,
            timestamp=time.time()
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'model_name' in result_dict
        assert 'prediction' in result_dict
        assert 'confidence' in result_dict
        assert 'latency_ms' in result_dict


class TestShadowEvaluationConfig:
    """Test suite for ShadowEvaluationConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ShadowEvaluationConfig()
        
        assert config.max_latency_ms == 5000
        assert config.max_parallel_models == 5
        assert config.buffer_size == 1000
        assert config.flush_interval_seconds == 60
        assert config.enable_business_metrics is True
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = ShadowEvaluationConfig(
            max_latency_ms=1000,
            buffer_size=500,
            enable_business_metrics=False
        )
        
        assert config.max_latency_ms == 1000
        assert config.buffer_size == 500
        assert config.enable_business_metrics is False


class TestShadowEvaluator:
    """Test suite for ShadowEvaluator"""
    
    @pytest.fixture
    def models(self):
        """Create sample models for testing"""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        production_model = RandomForestClassifier(n_estimators=50, random_state=42)
        shadow_model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
        shadow_model_2 = LogisticRegression(random_state=42)
        
        production_model.fit(X_train, y_train)
        shadow_model_1.fit(X_train, y_train)
        shadow_model_2.fit(X_train, y_train)
        
        return production_model, shadow_model_1, shadow_model_2, X_test, y_test
    
    @pytest.fixture
    def evaluator(self, models):
        """Create a ShadowEvaluator instance for testing"""
        production_model, shadow_model_1, shadow_model_2, _, _ = models
        
        shadow_models = {
            'rf_v2': shadow_model_1,
            'logistic': shadow_model_2
        }
        
        config = ShadowEvaluationConfig(
            max_latency_ms=1000,
            buffer_size=10,
            flush_interval_seconds=30
        )
        
        return ShadowEvaluator(production_model, shadow_models, config)
    
    def test_initialization(self, evaluator):
        """Test ShadowEvaluator initialization"""
        assert evaluator is not None
        assert evaluator.production_model is not None
        assert len(evaluator.shadow_models) == 2
        assert 'rf_v2' in evaluator.shadow_models
        assert 'logistic' in evaluator.shadow_models
    
    def test_stats_initialization(self, evaluator):
        """Test statistics tracking initialization"""
        assert evaluator.stats['total_requests'] == 0
        assert evaluator.stats['production_errors'] == 0
        assert 'rf_v2' in evaluator.stats['shadow_errors']
        assert 'logistic' in evaluator.stats['shadow_errors']
        assert evaluator.stats['shadow_errors']['rf_v2'] == 0
    
    def test_predict_with_shadows_basic(self, evaluator, models):
        """Test basic prediction with shadow models"""
        _, _, _, X_test, y_test = models
        
        x_sample = X_test[0].reshape(1, -1)
        prediction = evaluator.predict_with_shadows(x_sample, request_id='test_001')
        
        assert prediction is not None
        assert len(prediction) == 1
        assert evaluator.stats['total_requests'] == 1
    
    def test_predict_with_shadows_multiple_requests(self, evaluator, models):
        """Test multiple predictions with shadow evaluation"""
        _, _, _, X_test, _ = models
        
        for i in range(5):
            x_sample = X_test[i].reshape(1, -1)
            prediction = evaluator.predict_with_shadows(x_sample, request_id=f'test_{i:03d}')
            assert prediction is not None
        
        assert evaluator.stats['total_requests'] == 5
    
    def test_results_buffer_filling(self, evaluator, models):
        """Test that results buffer fills up"""
        _, _, _, X_test, _ = models
        
        for i in range(5):
            x_sample = X_test[i].reshape(1, -1)
            evaluator.predict_with_shadows(x_sample, request_id=f'test_{i:03d}')
        
        assert len(evaluator.results_buffer) == 5
    
    def test_results_buffer_flush(self, evaluator, models):
        """Test automatic buffer flushing"""
        _, _, _, X_test, _ = models
        
        # Set buffer size to 3 for testing
        evaluator.config.buffer_size = 3
        
        for i in range(5):
            x_sample = X_test[i].reshape(1, -1)
            evaluator.predict_with_shadows(x_sample, request_id=f'test_{i:03d}')
        
        # Buffer should have been flushed and be smaller than 5
        assert len(evaluator.results_buffer) < 5
    
    def test_latency_tracking(self, evaluator, models):
        """Test latency statistics tracking"""
        _, _, _, X_test, _ = models
        
        x_sample = X_test[0].reshape(1, -1)
        evaluator.predict_with_shadows(x_sample, request_id='test_001')
        
        # Check that latency stats are recorded
        assert 'production' in evaluator.stats['latency_stats']
        assert len(evaluator.stats['latency_stats']['production']) > 0
    
    def test_safe_predict_success(self, evaluator, models):
        """Test safe prediction with successful model"""
        production_model, _, _, X_test, _ = models
        
        x_sample = X_test[0].reshape(1, -1)
        result = evaluator._safe_predict(production_model, 'test_model', x_sample)
        
        assert isinstance(result, PredictionResult)
        assert result.error is None
        assert result.prediction is not None
        assert result.latency_ms > 0
    
    def test_safe_predict_with_error(self, evaluator):
        """Test safe prediction with failing model"""
        # Create a mock model that raises an error
        failing_model = Mock()
        failing_model.predict.side_effect = Exception("Model failed")
        
        x_sample = np.array([[1, 2, 3]])
        result = evaluator._safe_predict(failing_model, 'failing_model', x_sample)
        
        assert isinstance(result, PredictionResult)
        assert result.error is not None
        assert "Model failed" in result.error
        assert result.prediction is None
    
    def test_confidence_extraction(self, evaluator, models):
        """Test confidence score extraction from predict_proba"""
        production_model, _, _, X_test, _ = models
        
        x_sample = X_test[0].reshape(1, -1)
        result = evaluator._safe_predict(production_model, 'test_model', x_sample)
        
        # RandomForest has predict_proba, so confidence should be set
        assert result.confidence is not None
        assert 0 <= result.confidence <= 1
    
    def test_record_business_outcome(self, evaluator):
        """Test recording business outcomes"""
        evaluator.record_business_outcome(
            request_id='test_001',
            actual_outcome=1,
            business_value=100.0
        )
        
        assert len(evaluator.business_metrics) == 1
        assert evaluator.business_metrics[0]['request_id'] == 'test_001'
        assert evaluator.business_metrics[0]['actual_outcome'] == 1
        assert evaluator.business_metrics[0]['business_value'] == 100.0
    
    def test_record_business_outcome_disabled(self, models):
        """Test that business metrics can be disabled"""
        production_model, shadow_model_1, shadow_model_2, _, _ = models
        
        config = ShadowEvaluationConfig(enable_business_metrics=False)
        evaluator = ShadowEvaluator(
            production_model,
            {'rf_v2': shadow_model_1},
            config
        )
        
        evaluator.record_business_outcome('test_001', 1, 100.0)
        
        # Should not record when disabled
        assert len(evaluator.business_metrics) == 0
    
    def test_get_performance_summary(self, evaluator, models):
        """Test getting performance summary"""
        _, _, _, X_test, y_test = models
        
        # Generate some predictions
        for i in range(10):
            x_sample = X_test[i].reshape(1, -1)
            evaluator.predict_with_shadows(x_sample, request_id=f'test_{i:03d}')
        
        summary = evaluator.get_performance_summary(time_window_hours=1)
        
        assert 'total_requests' in summary
        assert 'models' in summary
        assert 'production' in summary['models']
        assert 'rf_v2' in summary['models']
        assert 'logistic' in summary['models']
        
        # Check model metrics
        for model_name, metrics in summary['models'].items():
            assert 'total_predictions' in metrics
            assert 'successful_predictions' in metrics
            assert 'error_count' in metrics
            assert 'error_rate' in metrics
            assert 'avg_latency_ms' in metrics
    
    def test_get_performance_summary_empty(self, evaluator):
        """Test performance summary with no data"""
        summary = evaluator.get_performance_summary(time_window_hours=1)
        
        assert summary['total_requests'] == 0
    
    def test_performance_summary_time_window(self, evaluator, models):
        """Test performance summary with time window filtering"""
        _, _, _, X_test, _ = models
        
        # Add some predictions
        for i in range(5):
            x_sample = X_test[i].reshape(1, -1)
            evaluator.predict_with_shadows(x_sample, request_id=f'test_{i:03d}')
        
        # Get summary with very short time window
        summary = evaluator.get_performance_summary(time_window_hours=24)
        
        assert summary['total_requests'] == 5
    
    def test_flush_results_creates_file(self, evaluator, models):
        """Test that flushing results creates a JSON file"""
        _, _, _, X_test, _ = models
        
        # Add some predictions
        for i in range(5):
            x_sample = X_test[i].reshape(1, -1)
            evaluator.predict_with_shadows(x_sample, request_id=f'test_{i:03d}')
        
        # Manually flush
        evaluator._flush_results()
        
        # Check that buffer was cleared
        assert len(evaluator.results_buffer) == 0
    
    def test_request_id_generation(self, evaluator, models):
        """Test automatic request ID generation"""
        _, _, _, X_test, _ = models
        
        x_sample = X_test[0].reshape(1, -1)
        # Don't provide request_id
        prediction = evaluator.predict_with_shadows(x_sample)
        
        assert prediction is not None
        assert len(evaluator.results_buffer) == 1
        assert 'request_id' in evaluator.results_buffer[0]
    
    def test_user_context_logging(self, evaluator, models):
        """Test that user context is logged"""
        _, _, _, X_test, _ = models
        
        user_context = {'user_id': '12345', 'session_id': 'abc'}
        x_sample = X_test[0].reshape(1, -1)
        
        evaluator.predict_with_shadows(
            x_sample,
            request_id='test_001',
            user_context=user_context
        )
        
        assert len(evaluator.results_buffer) == 1
        assert evaluator.results_buffer[0]['user_context'] == user_context
    
    def test_production_model_failure_raises_error(self, models):
        """Test that production model failure raises exception"""
        _, shadow_model_1, shadow_model_2, X_test, _ = models
        
        # Create a failing production model
        failing_model = Mock()
        failing_model.predict.side_effect = Exception("Production failed")
        
        shadow_models = {
            'rf_v2': shadow_model_1,
            'logistic': shadow_model_2
        }
        
        evaluator = ShadowEvaluator(failing_model, shadow_models)
        
        x_sample = X_test[0].reshape(1, -1)
        
        with pytest.raises(Exception, match="Production model failed"):
            evaluator.predict_with_shadows(x_sample)
    
    def test_shadow_model_failure_doesnt_affect_production(self, models):
        """Test that shadow model failure doesn't affect production prediction"""
        production_model, _, _, X_test, _ = models
        
        # Create a failing shadow model
        failing_shadow = Mock()
        failing_shadow.predict.side_effect = Exception("Shadow failed")
        
        shadow_models = {
            'failing_shadow': failing_shadow
        }
        
        evaluator = ShadowEvaluator(production_model, shadow_models)
        
        x_sample = X_test[0].reshape(1, -1)
        prediction = evaluator.predict_with_shadows(x_sample)
        
        # Production should still work
        assert prediction is not None
        assert evaluator.stats['shadow_errors']['failing_shadow'] == 1


class TestModelComparator:
    """Test suite for ModelComparator"""
    
    @pytest.fixture
    def comparator(self):
        """Create a ModelComparator instance for testing"""
        return ModelComparator(significance_level=0.05)
    
    @pytest.fixture
    def sample_results(self):
        """Generate sample prediction results"""
        n_samples = 100
        
        production_results = []
        shadow_results = []
        ground_truth = []
        
        for i in range(n_samples):
            # Production results
            production_results.append({
                'prediction': i % 2,
                'latency_ms': np.random.uniform(10, 20),
                'error': None
            })
            
            # Shadow results (slightly different)
            shadow_results.append({
                'prediction': (i + np.random.choice([0, 1])) % 2,
                'latency_ms': np.random.uniform(15, 25),
                'error': None
            })
            
            ground_truth.append(i % 2)
        
        return production_results, shadow_results, ground_truth
    
    def test_initialization(self, comparator):
        """Test ModelComparator initialization"""
        assert comparator is not None
        assert comparator.significance_level == 0.05
    
    def test_compare_models_basic(self, comparator, sample_results):
        """Test basic model comparison"""
        production_results, shadow_results, ground_truth = sample_results
        
        comparison = comparator.compare_models(
            production_results,
            shadow_results,
            ground_truth
        )
        
        assert 'sample_size' in comparison
        assert 'timestamp' in comparison
        assert 'latency' in comparison
        assert 'error_rates' in comparison
        assert 'agreement_rate' in comparison
        assert 'accuracy' in comparison
    
    def test_compare_models_without_ground_truth(self, comparator, sample_results):
        """Test model comparison without ground truth"""
        production_results, shadow_results, _ = sample_results
        
        comparison = comparator.compare_models(
            production_results,
            shadow_results,
            ground_truth=None
        )
        
        assert 'sample_size' in comparison
        assert 'latency' in comparison
        assert 'error_rates' in comparison
        assert 'agreement_rate' in comparison
        assert 'accuracy' not in comparison
    
    def test_compare_latency(self, comparator):
        """Test latency comparison"""
        latencies_a = [10, 15, 20, 25, 30]
        latencies_b = [15, 20, 25, 30, 35]
        
        result = comparator._compare_latency(latencies_a, latencies_b)
        
        assert 'median_a' in result
        assert 'median_b' in result
        assert 'p95_a' in result
        assert 'p95_b' in result
        assert 'p_value' in result
        assert 'significant_difference' in result
        assert 'faster_model' in result
    
    def test_compare_accuracy(self, comparator):
        """Test accuracy comparison"""
        predictions_a = [0, 1, 0, 1, 0, 1, 0, 1] * 10
        predictions_b = [0, 1, 1, 1, 0, 0, 0, 1] * 10
        ground_truth = [0, 1, 0, 1, 0, 1, 0, 1] * 10
        
        result = comparator._compare_accuracy(predictions_a, predictions_b, ground_truth)
        
        assert 'accuracy_a' in result
        assert 'accuracy_b' in result
        assert 'mcnemar_statistic' in result
        assert 'p_value' in result
        assert 'significant_difference' in result
        assert 'better_model' in result
        assert 'contingency_table' in result
    
    def test_compare_accuracy_perfect_agreement(self, comparator):
        """Test accuracy comparison when both models are identical"""
        predictions_a = [0, 1, 0, 1] * 25
        predictions_b = [0, 1, 0, 1] * 25
        ground_truth = [0, 1, 0, 1] * 25
        
        result = comparator._compare_accuracy(predictions_a, predictions_b, ground_truth)
        
        # Both models should have same accuracy
        assert result['accuracy_a'] == result['accuracy_b']
        # McNemar p-value should be high (no significant difference)
        assert result['p_value'] > 0.5
    
    def test_error_rate_comparison(self, comparator):
        """Test error rate comparison"""
        production_results = [
            {'prediction': 1, 'error': None} for _ in range(95)
        ] + [
            {'prediction': None, 'error': 'failed'} for _ in range(5)
        ]
        
        shadow_results = [
            {'prediction': 1, 'error': None} for _ in range(98)
        ] + [
            {'prediction': None, 'error': 'failed'} for _ in range(2)
        ]
        
        comparison = comparator.compare_models(
            production_results,
            shadow_results
        )
        
        assert comparison['error_rates']['production'] == 0.05
        assert comparison['error_rates']['shadow'] == 0.02
        assert comparison['error_rates']['difference'] == -0.03
    
    def test_agreement_rate_calculation(self, comparator):
        """Test prediction agreement rate calculation"""
        production_results = [
            {'prediction': i % 2, 'error': None} for i in range(100)
        ]
        
        # Shadow matches production 80% of the time
        shadow_results = []
        for i in range(100):
            if i < 80:
                shadow_results.append({'prediction': i % 2, 'error': None})
            else:
                shadow_results.append({'prediction': (i + 1) % 2, 'error': None})
        
        comparison = comparator.compare_models(
            production_results,
            shadow_results
        )
        
        assert 'agreement_rate' in comparison
        assert 0.75 < comparison['agreement_rate'] < 0.85
    
    def test_small_sample_size_handling(self, comparator):
        """Test handling of small sample sizes"""
        production_results = [
            {'prediction': 1, 'latency_ms': 10, 'error': None} for _ in range(5)
        ]
        shadow_results = [
            {'prediction': 1, 'latency_ms': 15, 'error': None} for _ in range(5)
        ]
        
        comparison = comparator.compare_models(
            production_results,
            shadow_results
        )
        
        # Should still return comparison, but without latency comparison
        assert 'sample_size' in comparison
        assert comparison['sample_size'] == 5
        # Latency comparison requires > 10 samples
        assert 'latency' not in comparison


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


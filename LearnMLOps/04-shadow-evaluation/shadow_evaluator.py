"""
Shadow Evaluation System
Safe model deployment with parallel evaluation and gradual rollout
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Container for model prediction results"""
    model_name: str
    prediction: Any
    confidence: Optional[float]
    latency_ms: float
    timestamp: float
    error: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ShadowEvaluationConfig:
    """Configuration for shadow evaluation"""
    max_latency_ms: float = 5000
    max_parallel_models: int = 5
    buffer_size: int = 1000
    flush_interval_seconds: int = 60
    enable_business_metrics: bool = True

class ShadowEvaluator:
    """
    Main shadow evaluation system for safe model deployment
    """
    
    def __init__(self, 
                 production_model: Any,
                 shadow_models: Dict[str, Any],
                 config: ShadowEvaluationConfig = None):
        
        self.production_model = production_model
        self.shadow_models = shadow_models
        self.config = config or ShadowEvaluationConfig()
        
        # Initialize components
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_parallel_models)
        self.results_buffer = []
        self.business_metrics = []
        self.last_flush = time.time()
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'production_errors': 0,
            'shadow_errors': {},
            'latency_stats': {}
        }
        
        # Initialize shadow error counters
        for model_name in self.shadow_models:
            self.stats['shadow_errors'][model_name] = 0
    
    def predict_with_shadows(self, X, request_id: str = None, 
                           user_context: Dict = None) -> Any:
        """
        Execute prediction with production model and shadow models in parallel
        """
        
        self.stats['total_requests'] += 1
        request_id = request_id or f"req_{int(time.time() * 1000)}"
        
        # Submit all model predictions to thread pool
        futures = {}
        
        # Production model (always runs)
        futures['production'] = self.executor.submit(
            self._safe_predict, self.production_model, 'production', X
        )
        
        # Shadow models (run in parallel)
        for model_name, model in self.shadow_models.items():
            futures[model_name] = self.executor.submit(
                self._safe_predict, model, model_name, X
            )
        
        # Collect results with timeout
        results = {}
        production_result = None
        
        for model_name, future in futures.items():
            try:
                result = future.result(timeout=self.config.max_latency_ms / 1000)
                results[model_name] = result
                
                if model_name == 'production':
                    production_result = result
                    
            except Exception as e:
                # Create error result
                error_result = PredictionResult(
                    model_name=model_name,
                    prediction=None,
                    confidence=None,
                    latency_ms=self.config.max_latency_ms,
                    timestamp=time.time(),
                    error=str(e)
                )
                results[model_name] = error_result
                
                # Track errors
                if model_name == 'production':
                    self.stats['production_errors'] += 1
                    production_result = error_result
                else:
                    self.stats['shadow_errors'][model_name] += 1
        
        # Log results for analysis
        self._log_shadow_results(results, X, request_id, user_context)
        
        # Handle production model failure
        if production_result is None or production_result.error:
            raise Exception(f"Production model failed: {production_result.error if production_result else 'Unknown error'}")
        
        return production_result.prediction
    
    def _safe_predict(self, model: Any, model_name: str, X) -> PredictionResult:
        """
        Safely execute model prediction with error handling and timing
        """
        start_time = time.time()
        
        try:
            # Make prediction
            prediction = model.predict(X)
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract confidence if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X)
                    if hasattr(proba, 'max'):
                        confidence = float(proba.max())
                    elif len(proba) > 0:
                        confidence = float(np.max(proba[0]))
                except:
                    pass
            
            return PredictionResult(
                model_name=model_name,
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms,
                timestamp=time.time()
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return PredictionResult(
                model_name=model_name,
                prediction=None,
                confidence=None,
                latency_ms=latency_ms,
                timestamp=time.time(),
                error=str(e)
            )
    
    def _log_shadow_results(self, results: Dict[str, PredictionResult], 
                           X, request_id: str, user_context: Dict = None):
        """
        Log shadow evaluation results for offline analysis
        """
        
        log_entry = {
            'timestamp': time.time(),
            'request_id': request_id,
            'input_features_hash': hash(str(X)) if X is not None else None,
            'user_context': user_context,
            'results': {name: result.to_dict() for name, result in results.items()}
        }
        
        self.results_buffer.append(log_entry)
        
        # Update latency statistics
        for model_name, result in results.items():
            if model_name not in self.stats['latency_stats']:
                self.stats['latency_stats'][model_name] = []
            
            if not result.error:
                self.stats['latency_stats'][model_name].append(result.latency_ms)
        
        # Flush buffer if needed
        if (len(self.results_buffer) >= self.config.buffer_size or 
            time.time() - self.last_flush > self.config.flush_interval_seconds):
            self._flush_results()
    
    def _flush_results(self):
        """
        Flush results buffer to persistent storage
        """
        if not self.results_buffer:
            return
        
        logger.info(f"Flushing {len(self.results_buffer)} shadow evaluation results")
        
        # In production, write to database, file system, or message queue
        # For demo, we'll just save to a JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shadow_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results_buffer, f, indent=2, default=str)
            
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        # Clear buffer
        self.results_buffer.clear()
        self.last_flush = time.time()
    
    def record_business_outcome(self, request_id: str, actual_outcome: Any, 
                               business_value: float = None):
        """
        Record business outcomes for shadow evaluation analysis
        """
        
        if not self.config.enable_business_metrics:
            return
        
        outcome_record = {
            'timestamp': time.time(),
            'request_id': request_id,
            'actual_outcome': actual_outcome,
            'business_value': business_value
        }
        
        self.business_metrics.append(outcome_record)
    
    def get_performance_summary(self, time_window_hours: int = 24) -> Dict:
        """
        Get performance summary for the specified time window
        """
        
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        # Filter recent results
        recent_results = [
            entry for entry in self.results_buffer 
            if entry['timestamp'] > cutoff_time
        ]
        
        summary = {
            'time_window_hours': time_window_hours,
            'total_requests': len(recent_results),
            'models': {}
        }
        
        # Analyze each model
        all_models = ['production'] + list(self.shadow_models.keys())
        
        for model_name in all_models:
            model_results = []
            
            for entry in recent_results:
                if model_name in entry['results']:
                    model_results.append(entry['results'][model_name])
            
            if model_results:
                # Calculate metrics
                successful_results = [r for r in model_results if not r.get('error')]
                error_count = len(model_results) - len(successful_results)
                
                latencies = [r['latency_ms'] for r in successful_results]
                
                summary['models'][model_name] = {
                    'total_predictions': len(model_results),
                    'successful_predictions': len(successful_results),
                    'error_count': error_count,
                    'error_rate': error_count / len(model_results) if model_results else 0,
                    'avg_latency_ms': np.mean(latencies) if latencies else None,
                    'p95_latency_ms': np.percentile(latencies, 95) if latencies else None,
                    'p99_latency_ms': np.percentile(latencies, 99) if latencies else None
                }
        
        return summary

class ModelComparator:
    """
    Statistical comparison between production and shadow models
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def compare_models(self, production_results: List[Dict], 
                      shadow_results: List[Dict], 
                      ground_truth: List[Any] = None) -> Dict:
        """
        Comprehensive comparison between production and shadow models
        """
        
        comparison = {
            'sample_size': len(production_results),
            'timestamp': time.time()
        }
        
        # Latency comparison
        prod_latencies = [r['latency_ms'] for r in production_results if not r.get('error')]
        shadow_latencies = [r['latency_ms'] for r in shadow_results if not r.get('error')]
        
        if len(prod_latencies) > 10 and len(shadow_latencies) > 10:
            comparison['latency'] = self._compare_latency(prod_latencies, shadow_latencies)
        
        # Error rate comparison
        prod_errors = sum(1 for r in production_results if r.get('error'))
        shadow_errors = sum(1 for r in shadow_results if r.get('error'))
        
        comparison['error_rates'] = {
            'production': prod_errors / len(production_results),
            'shadow': shadow_errors / len(shadow_results),
            'difference': (shadow_errors / len(shadow_results)) - (prod_errors / len(production_results))
        }
        
        # Prediction agreement
        prod_predictions = [r['prediction'] for r in production_results if not r.get('error')]
        shadow_predictions = [r['prediction'] for r in shadow_results if not r.get('error')]
        
        if len(prod_predictions) > 0 and len(shadow_predictions) > 0:
            min_length = min(len(prod_predictions), len(shadow_predictions))
            agreements = sum(
                1 for i in range(min_length) 
                if prod_predictions[i] == shadow_predictions[i]
            )
            comparison['agreement_rate'] = agreements / min_length
        
        # Accuracy comparison (if ground truth available)
        if ground_truth:
            comparison['accuracy'] = self._compare_accuracy(
                prod_predictions, shadow_predictions, ground_truth
            )
        
        return comparison
    
    def _compare_latency(self, latencies_a: List[float], 
                        latencies_b: List[float]) -> Dict:
        """
        Compare latency distributions using statistical tests
        """
        
        # Wilcoxon signed-rank test for paired samples
        try:
            statistic, p_value = stats.wilcoxon(latencies_a, latencies_b)
        except:
            # Fall back to Mann-Whitney U test for unpaired samples
            statistic, p_value = stats.mannwhitneyu(latencies_a, latencies_b)
        
        return {
            'median_a': np.median(latencies_a),
            'median_b': np.median(latencies_b),
            'p95_a': np.percentile(latencies_a, 95),
            'p95_b': np.percentile(latencies_b, 95),
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant_difference': p_value < self.significance_level,
            'faster_model': 'a' if np.median(latencies_a) < np.median(latencies_b) else 'b'
        }
    
    def _compare_accuracy(self, predictions_a: List, predictions_b: List, 
                         ground_truth: List) -> Dict:
        """
        Compare accuracy using McNemar's test
        """
        
        # Align predictions with ground truth
        min_length = min(len(predictions_a), len(predictions_b), len(ground_truth))
        
        correct_a = [predictions_a[i] == ground_truth[i] for i in range(min_length)]
        correct_b = [predictions_b[i] == ground_truth[i] for i in range(min_length)]
        
        # McNemar's test contingency table
        both_correct = sum(a and b for a, b in zip(correct_a, correct_b))
        a_correct_b_wrong = sum(a and not b for a, b in zip(correct_a, correct_b))
        a_wrong_b_correct = sum(not a and b for a, b in zip(correct_a, correct_b))
        both_wrong = sum(not a and not b for a, b in zip(correct_a, correct_b))
        
        # McNemar's test statistic
        if a_correct_b_wrong + a_wrong_b_correct > 0:
            mcnemar_stat = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / (a_correct_b_wrong + a_wrong_b_correct)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat = 0
            p_value = 1.0
        
        return {
            'accuracy_a': np.mean(correct_a),
            'accuracy_b': np.mean(correct_b),
            'mcnemar_statistic': float(mcnemar_stat),
            'p_value': float(p_value),
            'significant_difference': p_value < self.significance_level,
            'better_model': 'a' if np.mean(correct_a) > np.mean(correct_b) else 'b',
            'contingency_table': {
                'both_correct': both_correct,
                'a_correct_b_wrong': a_correct_b_wrong,
                'a_wrong_b_correct': a_wrong_b_correct,
                'both_wrong': both_wrong
            }
        }

def demo_shadow_evaluation():
    """
    Demonstrate shadow evaluation system
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("🔍 Shadow Evaluation Demo")
    print("=" * 50)
    
    # Create sample dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train models
    production_model = RandomForestClassifier(n_estimators=50, random_state=42)
    shadow_model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
    shadow_model_2 = LogisticRegression(random_state=42)
    
    production_model.fit(X_train, y_train)
    shadow_model_1.fit(X_train, y_train)
    shadow_model_2.fit(X_train, y_train)
    
    # Setup shadow evaluator
    shadow_models = {
        'rf_v2': shadow_model_1,
        'logistic': shadow_model_2
    }
    
    config = ShadowEvaluationConfig(
        max_latency_ms=1000,
        buffer_size=50,
        flush_interval_seconds=30
    )
    
    evaluator = ShadowEvaluator(production_model, shadow_models, config)
    
    print("\n🚀 Running shadow evaluation on test data...")
    
    # Run predictions with shadow evaluation
    predictions = []
    for i, (x_sample, y_true) in enumerate(zip(X_test[:100], y_test[:100])):
        request_id = f"test_req_{i}"
        
        try:
            pred = evaluator.predict_with_shadows(
                x_sample.reshape(1, -1), 
                request_id=request_id
            )
            predictions.append(pred[0])
            
            # Simulate business outcome recording
            evaluator.record_business_outcome(
                request_id=request_id,
                actual_outcome=y_true,
                business_value=np.random.uniform(10, 100)
            )
            
        except Exception as e:
            print(f"Error in prediction {i}: {e}")
    
    # Get performance summary
    print("\n📊 Performance Summary:")
    summary = evaluator.get_performance_summary(time_window_hours=1)
    
    for model_name, metrics in summary['models'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  Predictions: {metrics['total_predictions']}")
        print(f"  Success rate: {(1 - metrics['error_rate']):.3f}")
        print(f"  Avg latency: {metrics['avg_latency_ms']:.1f}ms")
        if metrics['p95_latency_ms']:
            print(f"  P95 latency: {metrics['p95_latency_ms']:.1f}ms")
    
    # Flush remaining results
    evaluator._flush_results()
    
    print(f"\n✅ Shadow evaluation completed!")
    print(f"Total requests processed: {evaluator.stats['total_requests']}")

if __name__ == "__main__":
    demo_shadow_evaluation()







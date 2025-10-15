# Shadow Evaluation & A/B Testing

> *"Deploy with confidence - test new models safely alongside production systems"*

## Overview

Shadow evaluation is a risk-free way to test new ML models in production by running them alongside your current model without affecting user experience. This guide shows you how to implement safe model deployment strategies.

## What is Shadow Evaluation?

Shadow evaluation (also called shadow mode or dark launching) involves:

1. **Production Model**: Serves real users and generates actual responses
2. **Shadow Model**: Processes the same requests but outputs are logged, not served
3. **Comparison System**: Analyzes performance differences between models

```
User Request
     ↓
┌─────────────┐    ┌─────────────┐
│ Production  │    │   Shadow    │
│   Model     │    │   Model     │
│  (serves)   │    │  (logs)     │
└─────────────┘    └─────────────┘
     ↓                    ↓
User Response         Log Analysis
```

## Benefits of Shadow Evaluation

### **Risk Mitigation**
- **Zero user impact** during testing
- **Real production data** validation
- **Gradual rollout** capability
- **Easy rollback** if issues arise

### **Better Insights**
- **Real-world performance** metrics
- **Edge case discovery** in production
- **Latency and resource** impact assessment
- **Business metric** correlation

### **Faster Deployment**
- **Parallel development** of multiple models
- **Continuous testing** without downtime
- **Data-driven decisions** for model updates

## Shadow Evaluation Architecture

### 1. **Basic Shadow Setup**

```python
class ShadowEvaluator:
    def __init__(self, production_model, shadow_model, logger):
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.logger = logger
        self.shadow_enabled = True
    
    def predict(self, X):
        # Production prediction (always runs)
        prod_prediction = self.production_model.predict(X)
        
        # Shadow prediction (runs in parallel)
        if self.shadow_enabled:
            try:
                shadow_prediction = self.shadow_model.predict(X)
                
                # Log both predictions for comparison
                self.logger.log_predictions(
                    input_data=X,
                    production_pred=prod_prediction,
                    shadow_pred=shadow_prediction,
                    timestamp=time.time()
                )
            except Exception as e:
                self.logger.log_error(f"Shadow model error: {e}")
        
        # Always return production prediction
        return prod_prediction
```

### 2. **Advanced Shadow System**

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class PredictionResult:
    model_name: str
    prediction: Any
    confidence: Optional[float]
    latency_ms: float
    error: Optional[str] = None

class AdvancedShadowEvaluator:
    def __init__(self, models: Dict[str, Any], primary_model: str):
        self.models = models
        self.primary_model = primary_model
        self.shadow_models = [name for name in models.keys() if name != primary_model]
        self.executor = ThreadPoolExecutor(max_workers=len(self.shadow_models) + 1)
        self.results_buffer = []
        
    def predict_with_shadow(self, X, request_id: str = None):
        \"\"\"
        Run primary model + all shadow models in parallel
        \"\"\"
        
        # Submit all predictions to thread pool
        futures = {}
        
        for model_name, model in self.models.items():
            future = self.executor.submit(self._safe_predict, model, model_name, X)
            futures[model_name] = future
        
        # Collect results
        results = {}
        for model_name, future in futures.items():
            try:
                results[model_name] = future.result(timeout=5.0)  # 5s timeout
            except Exception as e:
                results[model_name] = PredictionResult(
                    model_name=model_name,
                    prediction=None,
                    confidence=None,
                    latency_ms=5000,  # Timeout
                    error=str(e)
                )
        
        # Log results for analysis
        self._log_shadow_results(results, X, request_id)
        
        # Return primary model result
        primary_result = results[self.primary_model]
        if primary_result.error:
            raise Exception(f"Primary model failed: {primary_result.error}")
        
        return primary_result.prediction
    
    def _safe_predict(self, model, model_name: str, X) -> PredictionResult:
        \"\"\"Safely execute model prediction with timing\"\"\"
        start_time = time.time()
        
        try:
            prediction = model.predict(X)
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract confidence if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X)
                    confidence = float(proba.max())
                except:
                    pass
            
            return PredictionResult(
                model_name=model_name,
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return PredictionResult(
                model_name=model_name,
                prediction=None,
                confidence=None,
                latency_ms=latency_ms,
                error=str(e)
            )
    
    def _log_shadow_results(self, results: Dict[str, PredictionResult], 
                           X, request_id: str):
        \"\"\"Log results for offline analysis\"\"\"
        
        log_entry = {
            'timestamp': time.time(),
            'request_id': request_id,
            'input_hash': hash(str(X)),  # For privacy
            'results': {
                name: {
                    'prediction': str(result.prediction),
                    'confidence': result.confidence,
                    'latency_ms': result.latency_ms,
                    'error': result.error
                }
                for name, result in results.items()
            }
        }
        
        self.results_buffer.append(log_entry)
        
        # Flush buffer periodically
        if len(self.results_buffer) >= 100:
            self._flush_results()
    
    def _flush_results(self):
        \"\"\"Flush results to persistent storage\"\"\"
        # In practice, write to database, file, or message queue
        print(f"Flushing {len(self.results_buffer)} shadow evaluation results")
        self.results_buffer.clear()
```

## 📊 Performance Comparison Framework

### 1. **Statistical Significance Testing**

```python
import numpy as np
from scipy import stats
from typing import List, Tuple

class ModelComparator:
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def compare_accuracy(self, model_a_correct: List[bool], 
                        model_b_correct: List[bool]) -> Dict:
        \"\"\"
        Compare accuracy between two models using McNemar's test
        \"\"\"
        
        # Create contingency table
        both_correct = sum(a and b for a, b in zip(model_a_correct, model_b_correct))
        a_correct_b_wrong = sum(a and not b for a, b in zip(model_a_correct, model_b_correct))
        a_wrong_b_correct = sum(not a and b for a, b in zip(model_a_correct, model_b_correct))
        both_wrong = sum(not a and not b for a, b in zip(model_a_correct, model_b_correct))
        
        # McNemar's test
        if a_correct_b_wrong + a_wrong_b_correct == 0:
            p_value = 1.0
            statistic = 0.0
        else:
            statistic = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / (a_correct_b_wrong + a_wrong_b_correct)
            p_value = 1 - stats.chi2.cdf(statistic, 1)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'model_a_accuracy': np.mean(model_a_correct),
            'model_b_accuracy': np.mean(model_b_correct),
            'contingency_table': {
                'both_correct': both_correct,
                'a_correct_b_wrong': a_correct_b_wrong,
                'a_wrong_b_correct': a_wrong_b_correct,
                'both_wrong': both_wrong
            }
        }
    
    def compare_latency(self, model_a_latency: List[float], 
                       model_b_latency: List[float]) -> Dict:
        \"\"\"
        Compare latency between models using Wilcoxon signed-rank test
        \"\"\"
        
        # Remove pairs where either model failed (latency = None)
        paired_latencies = [(a, b) for a, b in zip(model_a_latency, model_b_latency) 
                           if a is not None and b is not None]
        
        if len(paired_latencies) < 10:
            return {'error': 'Insufficient data for latency comparison'}
        
        a_latencies, b_latencies = zip(*paired_latencies)
        
        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(a_latencies, b_latencies)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'model_a_median_latency': np.median(a_latencies),
            'model_b_median_latency': np.median(b_latencies),
            'model_a_p95_latency': np.percentile(a_latencies, 95),
            'model_b_p95_latency': np.percentile(b_latencies, 95)
        }
    
    def analyze_prediction_agreement(self, predictions_a: List, 
                                   predictions_b: List) -> Dict:
        \"\"\"
        Analyze how often two models agree on predictions
        \"\"\"
        
        agreements = [a == b for a, b in zip(predictions_a, predictions_b)]
        agreement_rate = np.mean(agreements)
        
        # Confidence interval for agreement rate
        n = len(agreements)
        se = np.sqrt(agreement_rate * (1 - agreement_rate) / n)
        ci_lower = agreement_rate - 1.96 * se
        ci_upper = agreement_rate + 1.96 * se
        
        return {
            'agreement_rate': agreement_rate,
            'confidence_interval': (ci_lower, ci_upper),
            'total_predictions': n,
            'agreements': sum(agreements)
        }
```

### 2. **Business Metrics Integration**

```python
class BusinessMetricsTracker:
    def __init__(self):
        self.metrics_buffer = []
    
    def track_business_outcome(self, request_id: str, model_name: str, 
                              prediction: Any, actual_outcome: Any = None,
                              business_value: float = None):
        \"\"\"
        Track business metrics for shadow evaluation
        \"\"\"
        
        entry = {
            'timestamp': time.time(),
            'request_id': request_id,
            'model_name': model_name,
            'prediction': prediction,
            'actual_outcome': actual_outcome,
            'business_value': business_value
        }
        
        self.metrics_buffer.append(entry)
    
    def calculate_business_impact(self, model_a: str, model_b: str, 
                                 metric_type: str = 'conversion_rate') -> Dict:
        \"\"\"
        Calculate business impact difference between models
        \"\"\"
        
        # Filter data for each model
        model_a_data = [entry for entry in self.metrics_buffer 
                       if entry['model_name'] == model_a]
        model_b_data = [entry for entry in self.metrics_buffer 
                       if entry['model_name'] == model_b]
        
        if metric_type == 'conversion_rate':
            # Assuming binary outcomes (converted/not converted)
            a_conversions = [entry['actual_outcome'] for entry in model_a_data 
                           if entry['actual_outcome'] is not None]
            b_conversions = [entry['actual_outcome'] for entry in model_b_data 
                           if entry['actual_outcome'] is not None]
            
            if len(a_conversions) == 0 or len(b_conversions) == 0:
                return {'error': 'Insufficient conversion data'}
            
            a_rate = np.mean(a_conversions)
            b_rate = np.mean(b_conversions)
            
            # Statistical test for conversion rates
            successes_a = sum(a_conversions)
            successes_b = sum(b_conversions)
            n_a = len(a_conversions)
            n_b = len(b_conversions)
            
            # Two-proportion z-test
            p_pooled = (successes_a + successes_b) / (n_a + n_b)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
            
            if se > 0:
                z_stat = (a_rate - b_rate) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = 0
                p_value = 1.0
            
            return {
                'model_a_conversion_rate': a_rate,
                'model_b_conversion_rate': b_rate,
                'difference': a_rate - b_rate,
                'relative_improvement': (a_rate - b_rate) / b_rate if b_rate > 0 else 0,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'sample_sizes': {'model_a': n_a, 'model_b': n_b}
            }
        
        elif metric_type == 'revenue':
            # Revenue per prediction
            a_revenues = [entry['business_value'] for entry in model_a_data 
                         if entry['business_value'] is not None]
            b_revenues = [entry['business_value'] for entry in model_b_data 
                         if entry['business_value'] is not None]
            
            if len(a_revenues) == 0 or len(b_revenues) == 0:
                return {'error': 'Insufficient revenue data'}
            
            # Mann-Whitney U test for revenue comparison
            statistic, p_value = stats.mannwhitneyu(a_revenues, b_revenues, 
                                                   alternative='two-sided')
            
            return {
                'model_a_mean_revenue': np.mean(a_revenues),
                'model_b_mean_revenue': np.mean(b_revenues),
                'model_a_median_revenue': np.median(a_revenues),
                'model_b_median_revenue': np.median(b_revenues),
                'difference': np.mean(a_revenues) - np.mean(b_revenues),
                'u_statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'sample_sizes': {'model_a': len(a_revenues), 'model_b': len(b_revenues)}
            }
```

## 🚀 Gradual Rollout Strategy

### 1. **Canary Deployment**

```python
class CanaryDeployment:
    def __init__(self, production_model, canary_model, initial_traffic_percent=5):
        self.production_model = production_model
        self.canary_model = canary_model
        self.traffic_percent = initial_traffic_percent
        self.success_metrics = []
        self.error_count = 0
        self.total_requests = 0
    
    def predict(self, X, user_id: str = None):
        \"\"\"
        Route traffic between production and canary models
        \"\"\"
        
        self.total_requests += 1
        
        # Determine which model to use
        if self._should_use_canary(user_id):
            try:
                prediction = self.canary_model.predict(X)
                self._record_success('canary')
                return prediction
            except Exception as e:
                self.error_count += 1
                self._record_error('canary', str(e))
                # Fallback to production model
                return self.production_model.predict(X)
        else:
            try:
                prediction = self.production_model.predict(X)
                self._record_success('production')
                return prediction
            except Exception as e:
                self._record_error('production', str(e))
                raise
    
    def _should_use_canary(self, user_id: str = None) -> bool:
        \"\"\"
        Determine if request should go to canary model
        \"\"\"
        
        # Simple percentage-based routing
        if user_id:
            # Consistent routing based on user ID
            user_hash = hash(user_id) % 100
            return user_hash < self.traffic_percent
        else:
            # Random routing
            return np.random.random() < (self.traffic_percent / 100)
    
    def increase_traffic(self, new_percent: int):
        \"\"\"
        Gradually increase canary traffic
        \"\"\"
        
        if self._is_canary_healthy():
            self.traffic_percent = min(new_percent, 100)
            print(f"Increased canary traffic to {self.traffic_percent}%")
        else:
            print("Canary model not healthy, keeping current traffic level")
    
    def _is_canary_healthy(self) -> bool:
        \"\"\"
        Check if canary model is performing well
        \"\"\"
        
        if self.total_requests < 100:  # Need minimum data
            return True
        
        error_rate = self.error_count / self.total_requests
        return error_rate < 0.01  # Less than 1% error rate
    
    def _record_success(self, model_type: str):
        self.success_metrics.append({
            'timestamp': time.time(),
            'model': model_type,
            'success': True
        })
    
    def _record_error(self, model_type: str, error: str):
        self.success_metrics.append({
            'timestamp': time.time(),
            'model': model_type,
            'success': False,
            'error': error
        })
```

## 📈 Monitoring and Alerting

### Key Metrics to Track

1. **Performance Metrics**
   - Accuracy/F1-score differences
   - Latency percentiles (p50, p95, p99)
   - Error rates
   - Resource utilization

2. **Business Metrics**
   - Conversion rates
   - Revenue per prediction
   - User engagement
   - Customer satisfaction

3. **System Metrics**
   - Request volume
   - Model availability
   - Data drift indicators
   - Feature distribution changes

### Alert Conditions

```python
class ShadowMonitor:
    def __init__(self, thresholds: Dict):
        self.thresholds = thresholds
        self.alerts = []
    
    def check_alerts(self, metrics: Dict):
        \"\"\"
        Check if any metrics exceed thresholds
        \"\"\"
        
        alerts = []
        
        # Performance degradation
        if metrics.get('accuracy_drop', 0) > self.thresholds.get('max_accuracy_drop', 0.05):
            alerts.append({
                'type': 'performance_degradation',
                'message': f"Accuracy dropped by {metrics['accuracy_drop']:.3f}",
                'severity': 'high'
            })
        
        # Latency increase
        if metrics.get('latency_increase', 0) > self.thresholds.get('max_latency_increase', 100):
            alerts.append({
                'type': 'latency_increase',
                'message': f"Latency increased by {metrics['latency_increase']:.1f}ms",
                'severity': 'medium'
            })
        
        # Error rate spike
        if metrics.get('error_rate', 0) > self.thresholds.get('max_error_rate', 0.01):
            alerts.append({
                'type': 'error_rate_spike',
                'message': f"Error rate: {metrics['error_rate']:.3f}",
                'severity': 'high'
            })
        
        return alerts
```

## 🎯 Best Practices

### ✅ **Do's**
- **Start small**: Begin with 1-5% shadow traffic
- **Monitor closely**: Set up comprehensive alerting
- **Test gradually**: Increase traffic slowly based on metrics
- **Plan rollback**: Have automated rollback triggers
- **Track business metrics**: Not just technical metrics

### ❌ **Don'ts**
- **Don't ignore latency**: Shadow models can impact user experience
- **Don't skip statistical tests**: Ensure differences are significant
- **Don't rush rollout**: Take time to validate thoroughly
- **Don't forget edge cases**: Test with diverse data
- **Don't ignore resource costs**: Shadow models consume resources

## 🚀 Next Steps

1. **Implement basic shadow evaluation** for your models
2. **Set up monitoring and alerting** systems
3. **Define success criteria** and rollback conditions
4. **Plan gradual rollout** strategy
5. **Integrate with CI/CD** pipeline for automated deployment

---

**💡 Pro Tip**: Shadow evaluation is not just about model performance - it's about building confidence in your ML systems. The investment in proper shadow testing pays off in reduced production incidents and faster innovation cycles!







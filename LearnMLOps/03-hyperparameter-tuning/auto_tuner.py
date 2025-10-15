"""
Automated Hyperparameter Tuning Pipeline
Complete implementation with multiple optimization strategies
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, 
    StratifiedKFold, KFold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
import xgboost as xgb
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

# Optional: Optuna for Bayesian optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

class AutoTuner:
    """
    Automated hyperparameter tuning with multiple strategies
    """
    
    def __init__(self, model_type='auto'):
        self.model_type = model_type
        self.best_params = None
        self.best_score = None
        self.tuning_history = []
        self.optimization_time = 0
        
        # Define parameter spaces for common models
        self.param_spaces = {
            'random_forest_clf': {
                'grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'random': {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(3, 20),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', None]
                },
                'bayesian': {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                    'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]}
                }
            },
            'xgboost_clf': {
                'grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'random': {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(3, 10),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4)
                },
                'bayesian': {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                    'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0}
                }
            },
            'logistic_regression': {
                'grid': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'random': {
                    'C': uniform(0.1, 100),
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'bayesian': {
                    'C': {'type': 'float', 'low': 0.1, 'high': 100, 'log': True},
                    'penalty': {'type': 'categorical', 'choices': ['l1', 'l2']},
                    'solver': {'type': 'categorical', 'choices': ['liblinear', 'saga']}
                }
            }
        }
    
    def detect_model_type(self, model):
        """Auto-detect model type for parameter space selection"""
        model_name = model.__class__.__name__.lower()
        
        if 'randomforest' in model_name:
            return 'random_forest_clf' if 'classifier' in model_name else 'random_forest_reg'
        elif 'xgb' in model_name or 'gradient' in model_name:
            return 'xgboost_clf' if 'classifier' in model_name else 'xgboost_reg'
        elif 'logistic' in model_name:
            return 'logistic_regression'
        elif 'svm' in model_name or 'svc' in model_name:
            return 'svm'
        else:
            return 'unknown'
    
    def tune_hyperparameters(self, model, X, y, strategy='bayesian', 
                           cv=5, n_trials=50, timeout=1800, scoring='accuracy'):
        """
        Main tuning function with multiple strategies
        """
        start_time = time.time()
        
        # Auto-detect model type if not specified
        if self.model_type == 'auto':
            self.model_type = self.detect_model_type(model)
        
        # Get parameter space
        if self.model_type in self.param_spaces:
            param_space = self.param_spaces[self.model_type].get(strategy, {})
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported. "
                           f"Available: {list(self.param_spaces.keys())}")
        
        if not param_space:
            raise ValueError(f"No parameter space defined for strategy '{strategy}'")
        
        # Choose tuning strategy
        if strategy == 'grid':
            result = self._grid_search(model, X, y, param_space, cv, scoring)
        elif strategy == 'random':
            result = self._random_search(model, X, y, param_space, cv, n_trials, scoring)
        elif strategy == 'bayesian':
            if not OPTUNA_AVAILABLE:
                print("Optuna not available, falling back to random search")
                param_space = self.param_spaces[self.model_type]['random']
                result = self._random_search(model, X, y, param_space, cv, n_trials, scoring)
            else:
                result = self._bayesian_optimization(model, X, y, param_space, cv, n_trials, timeout, scoring)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.optimization_time = time.time() - start_time
        
        return result
    
    def _grid_search(self, model, X, y, param_grid, cv, scoring):
        """Grid search implementation"""
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, 
            n_jobs=-1, return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        # Store detailed results
        results_df = pd.DataFrame(grid_search.cv_results_)
        self.tuning_history = results_df.to_dict('records')
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_model': grid_search.best_estimator_,
            'cv_results': results_df,
            'n_combinations': len(results_df)
        }
    
    def _random_search(self, model, X, y, param_dist, cv, n_iter, scoring):
        """Random search implementation"""
        
        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=n_iter, cv=cv, 
            scoring=scoring, n_jobs=-1, random_state=42,
            return_train_score=True
        )
        
        random_search.fit(X, y)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        # Store detailed results
        results_df = pd.DataFrame(random_search.cv_results_)
        self.tuning_history = results_df.to_dict('records')
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_model': random_search.best_estimator_,
            'cv_results': results_df,
            'n_combinations': len(results_df)
        }
    
    def _bayesian_optimization(self, model, X, y, param_space, cv, n_trials, timeout, scoring):
        """Bayesian optimization using Optuna"""
        
        def objective(trial):
            # Sample parameters based on their types
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            # Handle special cases for specific models
            if 'solver' in params and 'penalty' in params:
                # Logistic regression: l1 penalty only works with certain solvers
                if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
                    params['solver'] = 'liblinear'
            
            # Create and evaluate model
            try:
                test_model = model.__class__(**params, random_state=42)
                scores = cross_val_score(test_model, X, y, cv=cv, scoring=scoring)
                score = scores.mean()
                
                # Store trial information
                self.tuning_history.append({
                    'params': params.copy(),
                    'score': score,
                    'std': scores.std(),
                    'trial_number': trial.number
                })
                
                return score
                
            except Exception as e:
                # Return poor score for invalid parameter combinations
                return -1.0
        
        # Create and run study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Create best model
        best_model = model.__class__(**self.best_params, random_state=42)
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_model': best_model,
            'study': study,
            'n_trials': len(study.trials)
        }
    
    def compare_strategies(self, model, X, y, strategies=['grid', 'random', 'bayesian'],
                         cv=5, max_time_per_strategy=600):
        """
        Compare different tuning strategies
        """
        results = {}
        
        for strategy in strategies:
            print(f"\n🔄 Running {strategy} search...")
            
            try:
                if strategy == 'grid':
                    result = self.tune_hyperparameters(
                        model, X, y, strategy=strategy, cv=cv
                    )
                else:
                    result = self.tune_hyperparameters(
                        model, X, y, strategy=strategy, cv=cv,
                        timeout=max_time_per_strategy
                    )
                
                results[strategy] = {
                    'best_score': result['best_score'],
                    'best_params': result['best_params'],
                    'optimization_time': self.optimization_time,
                    'n_evaluations': result.get('n_combinations', result.get('n_trials', 0))
                }
                
                print(f"✅ {strategy}: Score={result['best_score']:.4f}, "
                      f"Time={self.optimization_time:.1f}s")
                
            except Exception as e:
                print(f"❌ {strategy} failed: {e}")
                results[strategy] = {'error': str(e)}
        
        return results
    
    def get_feature_importance_during_tuning(self, model, X, y):
        """
        Analyze feature importance across different hyperparameter settings
        """
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importance_history = []
        
        for trial in self.tuning_history:
            if 'params' in trial:
                # Create model with these parameters
                temp_model = model.__class__(**trial['params'], random_state=42)
                temp_model.fit(X, y)
                
                if hasattr(temp_model, 'feature_importances_'):
                    importance_history.append({
                        'trial': trial.get('trial_number', len(importance_history)),
                        'score': trial['score'],
                        'importances': temp_model.feature_importances_
                    })
        
        return importance_history

def demo_auto_tuning():
    """
    Demonstrate automated hyperparameter tuning
    """
    from sklearn.datasets import make_classification, load_wine
    from sklearn.model_selection import train_test_split
    
    print("🎯 Automated Hyperparameter Tuning Demo")
    print("=" * 50)
    
    # Load sample dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                              n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize tuner
    tuner = AutoTuner()
    
    # Test with Random Forest
    print("\n🌲 Tuning Random Forest Classifier:")
    rf_model = RandomForestClassifier(random_state=42)
    
    # Compare strategies (quick demo with limited trials)
    comparison = tuner.compare_strategies(
        rf_model, X_train, y_train,
        strategies=['random', 'bayesian'] if OPTUNA_AVAILABLE else ['random'],
        max_time_per_strategy=60  # 1 minute per strategy
    )
    
    print("\n📊 Strategy Comparison:")
    for strategy, result in comparison.items():
        if 'error' not in result:
            print(f"  {strategy.capitalize()}: "
                  f"Score={result['best_score']:.4f}, "
                  f"Time={result['optimization_time']:.1f}s, "
                  f"Evaluations={result['n_evaluations']}")
    
    # Get best model and evaluate on test set
    if comparison:
        best_strategy = max(comparison.keys(), 
                          key=lambda k: comparison[k].get('best_score', 0))
        
        print(f"\n🏆 Best strategy: {best_strategy}")
        
        # Retrain with best parameters and evaluate
        best_result = tuner.tune_hyperparameters(
            rf_model, X_train, y_train, strategy=best_strategy, n_trials=20
        )
        
        final_model = best_result['best_model']
        test_score = final_model.score(X_test, y_test)
        
        print(f"📈 Final test score: {test_score:.4f}")
        print(f"🔧 Best parameters: {best_result['best_params']}")

if __name__ == "__main__":
    demo_auto_tuning()







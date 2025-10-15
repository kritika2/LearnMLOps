# Hyperparameter Tuning in Practice

> *"The difference between a good model and a great model is often in the hyperparameters"*

## Overview

Hyperparameter tuning can make or break your model's performance. This guide provides practical, automated approaches to optimize your models efficiently without wasting computational resources.

## Understanding Hyperparameter Types

### 1. **Model-Specific Parameters**

| Algorithm | Key Hyperparameters | Impact | Tuning Priority |
|-----------|-------------------|---------|----------------|
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_split` | Performance, Overfitting | High |
| **XGBoost** | `learning_rate`, `max_depth`, `n_estimators`, `subsample` | Speed, Accuracy | High |
| **SVM** | `C`, `gamma`, `kernel` | Decision boundary | Medium |
| **Neural Networks** | `learning_rate`, `batch_size`, `hidden_layers` | Convergence, Capacity | High |
| **Logistic Regression** | `C`, `penalty` | Regularization | Low |

### 2. **Universal Parameters**
- **Regularization**: Controls overfitting (`alpha`, `C`, `lambda`)
- **Learning Rate**: Controls convergence speed
- **Complexity**: Controls model capacity (`max_depth`, `n_estimators`)

## Tuning Strategies Comparison

### 1. **Grid Search** 
**Best for**: Small parameter spaces, interpretable results  
**Avoid when**: High-dimensional spaces, expensive models

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
```

### 2. **Random Search**
**Best for**: High-dimensional spaces, quick exploration  
**Avoid when**: Need systematic coverage

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=100,  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
```

### 3. **Bayesian Optimization** 
**Best for**: Expensive models, intelligent exploration  
**Avoid when**: Simple problems, need interpretability

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    
    model = XGBClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## Resource-Efficient Tuning Strategies

### 1. **Progressive Tuning**
Start simple, then refine:

```python
# Stage 1: Coarse grid (fast)
coarse_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 7, None]
}

# Stage 2: Fine-tune around best (slower)
best_n_est = coarse_best['n_estimators']
fine_params = {
    'n_estimators': [best_n_est-20, best_n_est, best_n_est+20],
    'max_depth': [best_depth-1, best_depth, best_depth+1],
    'min_samples_split': [2, 5, 10]
}
```

### 2. **Early Stopping**
For iterative algorithms:

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=1000,
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    eval_metric='logloss'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
```

### 3. **Successive Halving**
Eliminate poor performers early:

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

halving_search = HalvingRandomSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    factor=3,  # Eliminate 2/3 of candidates each round
    cv=5,
    random_state=42
)
```

## Complete Tuning Pipeline

### Automated Hyperparameter Optimizer

```python
class HyperparameterOptimizer:
    def __init__(self, model, param_space, strategy='bayesian'):
        self.model = model
        self.param_space = param_space
        self.strategy = strategy
        self.best_params = None
        self.best_score = None
        self.tuning_history = []
    
    def optimize(self, X, y, cv=5, n_trials=100, timeout=3600):
        """
        Optimize hyperparameters using specified strategy
        """
        
        if self.strategy == 'grid':
            return self._grid_search(X, y, cv)
        elif self.strategy == 'random':
            return self._random_search(X, y, cv, n_trials)
        elif self.strategy == 'bayesian':
            return self._bayesian_optimization(X, y, cv, n_trials, timeout)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _bayesian_optimization(self, X, y, cv, n_trials, timeout):
        """Bayesian optimization using Optuna"""
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in self.param_space.items():
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
            
            # Train and evaluate model
            model = self.model.__class__(**params, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv)
            score = scores.mean()
            
            # Store history
            self.tuning_history.append({
                'params': params.copy(),
                'score': score,
                'std': scores.std()
            })
            
            return score
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials),
            'study': study
        }
    
    def get_tuned_model(self):
        """Return model with best parameters"""
        if self.best_params is None:
            raise ValueError("Must run optimization first")
        
        return self.model.__class__(**self.best_params, random_state=42)
```

## 📈 Advanced Tuning Techniques

### 1. **Multi-Objective Optimization**
Optimize for multiple metrics:

```python
def multi_objective(trial):
    params = {...}  # Sample parameters
    
    model = XGBClassifier(**params)
    
    # Multiple objectives
    accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    speed = -time_model_training(model, X, y)  # Negative because we want faster
    
    return accuracy, speed  # Return tuple for multi-objective

study = optuna.create_study(directions=['maximize', 'maximize'])
study.optimize(multi_objective, n_trials=100)
```

### 2. **Nested Cross-Validation**
Unbiased performance estimation:

```python
def nested_cv_tuning(model, param_space, X, y, outer_cv=5, inner_cv=3):
    """
    Nested CV: Outer loop for evaluation, inner loop for tuning
    """
    outer_scores = []
    
    for train_idx, test_idx in KFold(n_splits=outer_cv).split(X):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        
        # Inner CV for hyperparameter tuning
        grid_search = GridSearchCV(model, param_space, cv=inner_cv)
        grid_search.fit(X_train_outer, y_train_outer)
        
        # Evaluate on outer test set
        score = grid_search.score(X_test_outer, y_test_outer)
        outer_scores.append(score)
    
    return np.array(outer_scores)
```

### 3. **Automated Feature Engineering + Tuning**
Combine feature selection with hyperparameter tuning:

```python
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

# Create pipeline with feature selection + model
pipeline = Pipeline([
    ('feature_selection', SelectKBest()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Tune both feature selection and model parameters
param_grid = {
    'feature_selection__k': [5, 10, 15, 'all'],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7, None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
```

## Common Pitfalls to Avoid

### 1. **Data Leakage in Tuning**
```python
# Wrong: Tune on full dataset
grid_search.fit(X, y)

# Correct: Tune only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
grid_search.fit(X_train, y_train)
final_score = grid_search.score(X_test, y_test)
```

### 2. **Overfitting to Validation Set**
```python
# ❌ Wrong: Too many tuning iterations on same validation set
for i in range(1000):
    score = tune_and_evaluate(params[i])
    if score > best_score:
        best_params = params[i]

# ✅ Correct: Use nested CV or separate test set
```

### 3. **Ignoring Computational Budget**
```python
# ❌ Wrong: Unlimited search
param_grid = {
    'param1': range(1, 1000),  # 999 values
    'param2': range(1, 1000),  # 999 values
}  # Total: 998,001 combinations!

# ✅ Correct: Reasonable search space
param_grid = {
    'param1': [1, 10, 100],
    'param2': [0.1, 1.0, 10.0]
}  # Total: 9 combinations
```

## 🎯 Model-Specific Tuning Guides

### Random Forest
**Priority Order**: `n_estimators` → `max_depth` → `min_samples_split` → `min_samples_leaf`

### XGBoost
**Priority Order**: `learning_rate` + `n_estimators` → `max_depth` → `subsample` → `colsample_bytree`

### Neural Networks
**Priority Order**: `learning_rate` → `batch_size` → `hidden_layer_sizes` → `dropout`

## 🚀 Next Steps

1. **Start with simple grid search** for baseline
2. **Use Bayesian optimization** for expensive models
3. **Implement early stopping** for iterative algorithms
4. **Monitor computational budget** vs. performance gains

---

**💡 Pro Tip**: Spend 80% of your tuning budget on the most impactful parameters. Often, 2-3 key hyperparameters determine 90% of the performance difference!

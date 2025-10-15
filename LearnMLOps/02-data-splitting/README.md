# Training Data Split Optimization

> *"The right split ratio can make or break your model's performance"*

## Overview

Data splitting is more nuanced than the common 70/20/10 rule. This guide shows you how to optimize your train/validation/test splits based on your specific dataset and problem characteristics.

## The Science Behind Split Ratios

### Why 70/20/10 Isn't Always Optimal

The traditional 70/20/10 split assumes:
- Large dataset (>10K samples)
- Balanced classes
- Independent samples
- Stable data distribution

**Reality Check**: Most real-world datasets violate these assumptions!

## Dynamic Split Strategies

### 1. **Dataset Size-Based Splits**

| Dataset Size | Train | Validation | Test | Reasoning |
|-------------|-------|------------|------|-----------|
| **Tiny** (< 1K) | 60% | 20% | 20% | Need more test data for reliable evaluation |
| **Small** (1K-10K) | 70% | 15% | 15% | Standard approach with slight adjustment |
| **Medium** (10K-100K) | 80% | 10% | 10% | More training data improves performance |
| **Large** (100K-1M) | 85% | 10% | 5% | Diminishing returns from test data |
| **Huge** (>1M) | 90% | 8% | 2% | Fixed validation/test sizes are sufficient |

### 2. **Problem-Specific Splits**

#### **Time Series Data**
```python
# Wrong: Random split destroys temporal order
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Correct: Temporal split preserves order
split_date = data.index[int(len(data) * 0.8)]
train = data[data.index <= split_date]
test = data[data.index > split_date]
```

#### **Imbalanced Classes**
```python
# Use stratified splitting to maintain class distribution
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,  # Maintains class proportions
    random_state=42
)
```

#### **Small Datasets**
```python
# Use cross-validation instead of fixed splits
from sklearn.model_selection import cross_val_score

# No separate test set - use CV for evaluation
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Score: {scores.mean():.3f} ± {scores.std():.3f}")
```

## Advanced Splitting Techniques

### 1. **Nested Cross-Validation**
For hyperparameter tuning + unbiased evaluation:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

# Outer loop: Model evaluation
# Inner loop: Hyperparameter tuning
outer_scores = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    # Inner CV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv)
    grid_search.fit(X_train_outer, y_train_outer)
    
    # Evaluate best model on outer test set
    score = grid_search.score(X_test_outer, y_test_outer)
    outer_scores.append(score)
```

### 2. **Group-Based Splitting**
When samples are not independent:

```python
from sklearn.model_selection import GroupShuffleSplit

# Example: Patient data where multiple samples per patient
groups = df['patient_id']  # Group identifier

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
```

### 3. **Time-Aware Validation**
For time series with concept drift:

```python
from sklearn.model_selection import TimeSeriesSplit

# Rolling window validation
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train on past, validate on future
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
```

## Practical Implementation

### Smart Split Calculator

```python
def calculate_optimal_split(dataset_size, problem_type, has_groups=False):
    \"\"\"
    Calculate optimal train/validation/test split ratios
    \"\"\"
    
    if dataset_size < 1000:
        # Small dataset - use cross-validation
        return {
            'strategy': 'cross_validation',
            'cv_folds': 5,
            'note': 'Too small for separate test set'
        }
    
    elif dataset_size < 10000:
        # Medium dataset
        if problem_type == 'time_series':
            return {'train': 0.7, 'val': 0.2, 'test': 0.1, 'temporal': True}
        else:
            return {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    elif dataset_size < 100000:
        # Large dataset
        return {'train': 0.8, 'val': 0.1, 'test': 0.1}
    
    else:
        # Very large dataset
        return {'train': 0.9, 'val': 0.08, 'test': 0.02}

# Example usage
split_config = calculate_optimal_split(
    dataset_size=5000,
    problem_type='classification'
)
print(f"Recommended split: {split_config}")
```

## Common Pitfalls to Avoid

### 1. **Data Leakage**
```python
# Wrong: Scaling before splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, y)

# Correct: Split first, then scale
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, don't fit!
```

### 2. **Temporal Leakage**
```python
# Wrong: Future data in training
df_shuffled = df.sample(frac=1)  # Random shuffle
train = df_shuffled[:800]
test = df_shuffled[800:]

# Correct: Respect temporal order
train = df[df['date'] < '2023-01-01']
test = df[df['date'] >= '2023-01-01']
```

### 3. **Group Leakage**
```python
# Wrong: Same patient in train and test
train_test_split(X, y, test_size=0.2)

# Correct: Patient-level splitting
unique_patients = df['patient_id'].unique()
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2)
```

## Validation Strategies by Use Case

| Use Case | Recommended Strategy | Key Considerations |
|----------|---------------------|-------------------|
| **Image Classification** | Stratified Split | Maintain class balance |
| **Time Series Forecasting** | Temporal Split | No future data in training |
| **Medical Diagnosis** | Group Split | Patient-level separation |
| **A/B Testing** | Stratified Split | Balance treatment groups |
| **Fraud Detection** | Stratified + Temporal | Handle class imbalance + time |
| **Recommendation Systems** | User-based Split | User-level separation |

## Monitoring Split Quality

### Key Metrics to Track

```python
def evaluate_split_quality(X_train, X_val, X_test, y_train, y_val, y_test):
    \"\"\"
    Evaluate the quality of your data split
    \"\"\"
    
    metrics = {}
    
    # Size distribution
    total_size = len(X_train) + len(X_val) + len(X_test)
    metrics['train_ratio'] = len(X_train) / total_size
    metrics['val_ratio'] = len(X_val) / total_size
    metrics['test_ratio'] = len(X_test) / total_size
    
    # Class distribution (for classification)
    if hasattr(y_train, 'value_counts'):
        train_dist = y_train.value_counts(normalize=True)
        val_dist = y_val.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        
        # Calculate distribution similarity (KL divergence could be used)
        metrics['class_balance_maintained'] = (
            abs(train_dist - val_dist).max() < 0.05 and
            abs(train_dist - test_dist).max() < 0.05
        )
    
    # Feature distribution similarity
    from scipy.stats import ks_2samp
    
    feature_similarity = []
    for i in range(X_train.shape[1]):
        _, p_val = ks_2samp(X_train[:, i], X_test[:, i])
        feature_similarity.append(p_val > 0.05)  # Similar if p > 0.05
    
    metrics['feature_distribution_similar'] = np.mean(feature_similarity) > 0.8
    
    return metrics
```

## Next Steps

1. **Analyze your dataset** characteristics
2. **Choose appropriate split strategy** based on problem type
3. **Implement proper validation** to avoid data leakage
4. **Monitor split quality** with the provided metrics

---

**Pro Tip**: When in doubt, use more conservative splits with larger test sets. It's better to be confident in your model's performance than to be surprised in production!







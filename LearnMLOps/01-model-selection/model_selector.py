"""
Model Selection Helper Tool
A practical implementation for systematic model selection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ModelSelector:
    """
    Automated model selection based on data characteristics
    """
    
    def __init__(self):
        self.supervised_models = {
            'regression': {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'SVR': SVR()
            },
            'classification': {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(random_state=42),
                'Naive Bayes': GaussianNB()
            }
        }
        
        self.unsupervised_models = {
            'clustering': {
                'K-Means': KMeans(n_clusters=3, random_state=42),
                'DBSCAN': DBSCAN()
            },
            'anomaly_detection': {
                'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
                'Local Outlier Factor': LocalOutlierFactor(contamination=0.1),
                'One-Class SVM': OneClassSVM(gamma='scale')
            },
            'dimensionality_reduction': {
                'PCA': PCA(n_components=2)
            }
        }
    
    def analyze_data(self, X, y=None):
        """
        Analyze dataset characteristics to guide model selection
        """
        n_samples, n_features = X.shape
        
        analysis = {
            'n_samples': n_samples,
            'n_features': n_features,
            'dataset_size': self._categorize_size(n_samples),
            'feature_density': 'high' if n_features > n_samples else 'normal',
            'has_labels': y is not None
        }
        
        if y is not None:
            if self._is_continuous(y):
                analysis['problem_type'] = 'regression'
                analysis['target_type'] = 'continuous'
            else:
                analysis['problem_type'] = 'classification'
                analysis['target_type'] = 'categorical'
                analysis['n_classes'] = len(np.unique(y))
        else:
            analysis['problem_type'] = 'unsupervised'
            analysis['target_type'] = 'unknown'
        
        return analysis
    
    def suggest_models(self, X, y=None, top_k=3):
        """
        Suggest top-k models based on data analysis
        """
        analysis = self.analyze_data(X, y)
        suggestions = []
        
        if analysis['has_labels']:
            # Supervised learning
            model_pool = self.supervised_models[analysis['problem_type']]
            
            # Rule-based suggestions
            if analysis['dataset_size'] == 'small':
                if analysis['problem_type'] == 'regression':
                    suggestions = ['Linear Regression', 'SVR']
                else:
                    suggestions = ['Logistic Regression', 'Naive Bayes']
            elif analysis['dataset_size'] == 'medium':
                suggestions = ['Random Forest', 'SVM']
            else:  # large
                suggestions = ['Random Forest']
                
        else:
            # Unsupervised learning - need to determine goal
            if analysis['feature_density'] == 'high' or analysis['n_samples'] > 10000:
                # Large/high-dimensional data - good for anomaly detection
                suggestions = ['Isolation Forest', 'PCA', 'K-Means']
            else:
                # Standard unsupervised tasks
                suggestions = ['K-Means', 'PCA', 'Isolation Forest']
        
        return suggestions[:top_k], analysis
    
    def evaluate_models(self, X, y, cv=5):
        """
        Evaluate multiple models using cross-validation
        """
        analysis = self.analyze_data(X, y)
        
        if not analysis['has_labels']:
            print("Cannot evaluate unsupervised models with cross-validation")
            return None
        
        model_pool = self.supervised_models[analysis['problem_type']]
        results = {}
        
        # Scale features for SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for name, model in model_pool.items():
            try:
                if 'SVM' in name or 'SVR' in name:
                    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy' if analysis['problem_type'] == 'classification' else 'r2')
                else:
                    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy' if analysis['problem_type'] == 'classification' else 'r2')
                
                results[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores
                }
            except Exception as e:
                results[name] = {
                    'mean_score': 0,
                    'std_score': 0,
                    'error': str(e)
                }
        
        return results
    
    def _categorize_size(self, n_samples):
        """Categorize dataset size"""
        if n_samples < 1000:
            return 'small'
        elif n_samples < 100000:
            return 'medium'
        else:
            return 'large'
    
    def _is_continuous(self, y):
        """Check if target variable is continuous"""
        return len(np.unique(y)) > 10 and np.issubdtype(y.dtype, np.number)

def demo_model_selection():
    """
    Demonstrate model selection on sample datasets
    """
    from sklearn.datasets import make_classification, make_regression, make_blobs
    
    print("🎯 Model Selection Demo")
    print("=" * 50)
    
    selector = ModelSelector()
    
    # Classification example
    print("\n📊 Classification Problem:")
    X_clf, y_clf = make_classification(n_samples=1000, n_features=10, n_classes=3, random_state=42)
    
    suggestions, analysis = selector.suggest_models(X_clf, y_clf)
    print(f"Dataset: {analysis['n_samples']} samples, {analysis['n_features']} features")
    print(f"Problem type: {analysis['problem_type']}")
    print(f"Suggested models: {suggestions}")
    
    # Evaluate models
    results = selector.evaluate_models(X_clf, y_clf)
    print("\nModel Performance:")
    for model, metrics in results.items():
        if 'error' not in metrics:
            print(f"  {model}: {metrics['mean_score']:.3f} ± {metrics['std_score']:.3f}")
        else:
            print(f"  {model}: Error - {metrics['error']}")
    
    # Regression example
    print("\n📈 Regression Problem:")
    X_reg, y_reg = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)
    
    suggestions, analysis = selector.suggest_models(X_reg, y_reg)
    print(f"Dataset: {analysis['n_samples']} samples, {analysis['n_features']} features")
    print(f"Problem type: {analysis['problem_type']}")
    print(f"Suggested models: {suggestions}")
    
    # Clustering example
    print("\n🔍 Clustering Problem:")
    X_cluster, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
    
    suggestions, analysis = selector.suggest_models(X_cluster)
    print(f"Dataset: {analysis['n_samples']} samples, {analysis['n_features']} features")
    print(f"Problem type: {analysis['problem_type']}")
    print(f"Suggested approaches: {suggestions}")
    
    # Anomaly Detection example
    print("\n🚨 Anomaly Detection Problem:")
    from sklearn.datasets import make_blobs
    
    # Create normal data with some outliers
    X_normal, _ = make_blobs(n_samples=200, centers=1, n_features=5, random_state=42)
    # Add some outliers
    X_outliers = np.random.uniform(low=-10, high=10, size=(20, 5))
    X_anomaly = np.vstack([X_normal, X_outliers])
    
    suggestions, analysis = selector.suggest_models(X_anomaly)
    print(f"Dataset: {analysis['n_samples']} samples, {analysis['n_features']} features")
    print(f"Problem type: {analysis['problem_type']}")
    print(f"Suggested approaches: {suggestions}")
    
    # Demonstrate Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_scores = iso_forest.fit_predict(X_anomaly)
    outliers_detected = np.sum(anomaly_scores == -1)
    print(f"Isolation Forest detected {outliers_detected} outliers (expected ~20)")

if __name__ == "__main__":
    demo_model_selection()

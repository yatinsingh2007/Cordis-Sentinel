import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score
)

def train_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str = 'logistic') -> Any:
    """
    Train a machine learning model.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target array
        model_type: Type of model to train ('logistic' or 'decision_tree')
        
    Returns:
        Trained sklearn model
    """
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42, max_depth=5)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'logistic' or 'decision_tree'.")
        
    print(f"Training {model_type}...")
    model.fit(X_train, y_train)
    print("Training complete.")
    
    return model

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained sklearn model
        X_test: Testing feature matrix
        y_test: Testing target array
        
    Returns:
        Tuple containing:
        - Dictionary of evaluation metrics
        - Array of predictions
        - Array of prediction probabilities
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'ROC_AUC': roc_auc
    }
    
    print("Model Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name:10s} : {value:.4f}")
        
    return metrics, y_pred, y_prob

def save_model(model: Any, path: str) -> None:
    """
    Save the trained model to disk using joblib.
    
    Args:
        model: Trained model to save
        path: File path to save the model to
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        path: File path to load the model from
        
    Returns:
        Loaded model
    """
    print(f"Loading model from {path}")
    return joblib.load(path)

if __name__ == "__main__":
    import os
    # Example standalone execution
    # This requires features_engineered.csv to exist
    data_path = '../data/processed/features_engineered.csv'
    model_save_path = '../models/logistic_regression_model.pkl'
    
    if os.path.exists(data_path):
        from sklearn.model_selection import train_test_split
        
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        TARGET = 'heart_attack_risk'
        X = df.drop(columns=[TARGET])
        y = df[TARGET]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # For a full test we would use SMOTE here, but for module testing we'll just train
        model = train_model(X_train, y_train, model_type='logistic')
        metrics, preds, probs = evaluate_model(model, X_test, y_test)
        
        save_model(model, model_save_path)
        print("Model testing complete.")
    else:
        print(f"Processed data file not found at {data_path}.")

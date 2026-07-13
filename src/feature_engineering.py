import pandas as pd
import numpy as np
import warnings
from typing import Tuple, List

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings('ignore')

def bp_category(bp: float) -> str:
    """Helper for BP category."""
    if bp < 120:
        return 'Normal'
    elif bp < 130:
        return 'Elevated'
    elif bp < 140:
        return 'Stage1'
    else:
        return 'Stage2'

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply manual feature engineering logic.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        DataFrame with new engineered features
    """
    df = df.copy()
    
    # Drop patient_id if it exists
    if 'patient_id' in df.columns:
        df.drop(columns=['patient_id'], inplace=True)
        
    # Age Group (ordinal bins)
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 40, 50, 60, 70, 120],
        labels=['<40', '40-49', '50-59', '60-69', '70+']
    ).astype(str)
    
    # Blood Pressure Category
    df['bp_category'] = df['resting_bp'].apply(bp_category)
    
    # Heart Rate Age Ratio
    df['hr_age_ratio'] = df['max_heart_rate'] / (220 - df['age'])
    
    # Heart Rate Reserve
    df['heart_rate_reserve'] = df['max_heart_rate'] - 60
    
    # Cholesterol BP Ratio
    df['cholesterol_bp_ratio'] = df['cholesterol'] / (df['resting_bp'] + 1e-5)
    
    # Lifestyle Risk Score
    smoking_map  = {'Never': 0, 'Former': 1, 'Current': 2}
    alcohol_map  = {'Non-drinker': 0, 'Moderate': 1, 'Heavy': 2}
    activity_map = {'High': 0, 'Moderate': 1, 'Low': 2}
    
    df['lifestyle_risk_score'] = (
        df['smoking_status'].map(smoking_map).fillna(0) +
        df['alcohol_consumption'].map(alcohol_map).fillna(0) +
        df['physical_activity'].map(activity_map).fillna(0)
    )
    
    # Comorbidity Score
    df['comorbidity_score'] = (
        df['diabetes'] +
        df['family_history'] +
        (df['stress_level'] >= 7).astype(int)
    )
    
    return df

def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build and fit a ColumnTransformer pipeline for encoding and scaling.
    
    Args:
        X: Input pandas DataFrame containing features
        
    Returns:
        Fitted ColumnTransformer object
    """
    ordinal_config = {
        'smoking_status'     : ['Never', 'Former', 'Current'],
        'alcohol_consumption': ['Non-drinker', 'Moderate', 'Heavy'],
        'physical_activity'  : ['Low', 'Moderate', 'High'],
        'st_slope'           : ['Down', 'Flat', 'Up'],
        'age_group'          : ['<40', '40-49', '50-59', '60-69', '70+'],
        'bp_category'        : ['Normal', 'Elevated', 'Stage1', 'Stage2'],
    }
    
    ordinal_cols = list(ordinal_config.keys())
    ordinal_categories = [ordinal_config[c] for c in ordinal_cols]
    
    # Only keep columns that exist in X
    ordinal_cols = [c for c in ordinal_cols if c in X.columns]
    ordinal_categories = [ordinal_config[c] for c in ordinal_cols]
    
    nominal_cols = ['gender', 'chest_pain_type', 'resting_ecg', 'thalassemia']
    nominal_cols = [c for c in nominal_cols if c in X.columns]
    
    scale_cols = [
        'age', 'resting_bp', 'cholesterol', 'max_heart_rate', 'oldpeak',
        'bmi', 'stress_level', 'hr_age_ratio', 'heart_rate_reserve',
        'cholesterol_bp_ratio', 'lifestyle_risk_score', 'comorbidity_score'
    ]
    scale_cols = [c for c in scale_cols if c in X.columns]
    
    # Passthrough columns (e.g. binary flags like diabetes that are already 0/1)
    # We can just ignore them and remainder='passthrough' will keep them
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', OrdinalEncoder(categories=ordinal_categories, dtype=int), ordinal_cols),
            ('nom', OneHotEncoder(drop='first', sparse_output=False, dtype=int), nominal_cols),
            ('num', StandardScaler(), scale_cols)
        ],
        remainder='passthrough'
    )
    
    # Fit the preprocessor
    preprocessor.fit(X)
    return preprocessor

def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE for class imbalance.
    
    Args:
        X: Feature matrix
        y: Target array
        random_state: Seed for reproducibility
        
    Returns:
        Resampled feature matrix and target array
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Dataset shape after SMOTE: {X_res.shape[0]} samples")
    return X_res, y_res

if __name__ == "__main__":
    import os
    # Example standalone execution
    input_path = '../data/interim/heart_attack_dataset_cleaned.csv'
    
    if os.path.exists(input_path):
        print(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # 1. Feature Engineering
        df = engineer_features(df)
        print("Engineered features shape:", df.shape)
        
        # 2. Build Pipeline
        TARGET = 'heart_attack_risk'
        if TARGET in df.columns:
            X = df.drop(columns=[TARGET])
            y = df[TARGET]
        else:
            X = df
            y = None
            
        preprocessor = build_preprocessing_pipeline(X)
        
        # We can extract transformed dataframe if needed
        # We will use get_feature_names_out to keep column names
        X_trans = preprocessor.transform(X)
        
        feature_names = preprocessor.get_feature_names_out()
        # Clean feature names (remove transformer prefixes like 'ord__', 'nom__')
        clean_names = [name.split('__')[-1] for name in feature_names]
        
        X_df = pd.DataFrame(X_trans, columns=clean_names, index=X.index)
        
        if y is not None:
            # 3. Handle Imbalance
            X_res, y_res = apply_smote(X_df, y)
            
        print("Feature engineering pipeline test complete.")
    else:
        print(f"Cleaned data file not found at {input_path}.")

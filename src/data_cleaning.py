import pandas as pd
import numpy as np
from typing import Tuple

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataframe.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        DataFrame with duplicates removed
    """
    df = df.copy()
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"Dropped {initial_rows - df.shape[0]} duplicate rows.")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values: median for numeric, mode for categorical.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        DataFrame with imputed missing values
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_cols = num_cols.drop(['patient_id', 'heart_attack_risk'], errors='ignore')
    
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed {col} with median: {median_val}")
            
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Imputed {col} with mode: {mode_val}")
            
    return df

def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns to category dtype.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        DataFrame with updated dtypes
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].astype('category')
    return df

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap outliers using the IQR method for specific columns.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        DataFrame with capped outliers
    """
    df = df.copy()
    outlier_columns = ['resting_bp', 'cholesterol', 'max_heart_rate', 'bmi']
    
    for column in outlier_columns:
        if column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            
            df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
            df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
            
            print(f"Capped {outliers_count} outliers in {column}")
            
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrator function to clean the data by applying all steps in sequence.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        Cleaned pandas DataFrame
    """
    print("Starting data cleaning...")
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = fix_dtypes(df)
    df = handle_outliers(df)
    
    assert df.isnull().sum().sum() == 0, "There are still missing values!"
    print("Data cleaning complete.")
    return df

if __name__ == "__main__":
    import os
    # Example standalone execution
    raw_data_path = '../data/raw/heart_attack_dataset_raw.csv'
    output_path = '../data/interim/heart_attack_dataset_cleaned.csv'
    
    if os.path.exists(raw_data_path):
        print(f"Loading data from {raw_data_path}")
        df_raw = pd.read_csv(raw_data_path)
        df_cleaned = clean_data(df_raw)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to: {output_path}")
    else:
        print(f"Raw data file not found at {raw_data_path}. Run from appropriate directory.")

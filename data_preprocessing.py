import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_dataframe(df, scale_numeric=True):
    """
    Converts categorical columns to numeric and optionally standardizes numeric columns.

    Parameters:
        df (pd.DataFrame): Input dataframe
        scale_numeric (bool): Whether to standardize numeric columns

    Returns:
        pd.DataFrame: Processed dataframe ready for ML
    """

    df_processed = df.copy()
    drop_cols = ["Employee ID"]

    # Separate numeric and categorical columns
    numeric_cols = [c for c in df_processed.select_dtypes(include=["number"]).columns if c not in drop_cols]
    categorical_cols = [c for c in df_processed.select_dtypes(exclude=["number"]).columns if c not in drop_cols]

    # Standardize numeric columns
    if scale_numeric and len(numeric_cols) > 0:
        scaler = StandardScaler()
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    # One-hot encode categorical columns
    if len(categorical_cols) > 0:
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=False) # Change to drop_first=True if using logistic regression model

    return df_processed
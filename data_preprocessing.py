import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from feature_engineering import FeatureEngineer

def preprocess_dataframe(df, scale_numeric=True):
    """
    Uses FeatureEngineer for static transformations, then optionally scales numeric
    columns and one‑hot encodes categorical columns.
    """

    # Apply feature engineering
    f = FeatureEngineer()
    df_processed = f.fit_transform(df) 
    return df_processed
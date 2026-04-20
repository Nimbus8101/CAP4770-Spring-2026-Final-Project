from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from feature_engineering import FeatureEngineer
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Logistic Regression pipeline
def build_lr_pipeline(model=None, numeric_cols=None, categorical_cols=None):
    if model is None:
        model = LogisticRegression(
            solver="saga",
            C=0.1,
            max_iter=2000
        )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02), categorical_cols)
        ]
    )
    
    return Pipeline(steps=[
        ("features", FeatureEngineer()),
        ("preprocess", preprocess),
        ("model", model)
    ])


# Random Forest pipeline
def build_rf_pipeline(model=None, numeric_cols=None, categorical_cols=None):
    if model is None:
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )

    preprocess = ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=0.02,
            sparse_output=True
        ), categorical_cols)
    ])
    
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocess", preprocess),
        ("model", model)
    ])
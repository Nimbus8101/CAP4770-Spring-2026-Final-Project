import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.clip_bounds = {}

    def fit(self, X, y=None):
        X = X.copy()

        # Precompute engineered columns needed for clipping
        # stagnation = X["Years at Company"] - X["Company Tenure"]
        # promotion_rate = X["Number of Promotions"] / (X["Years at Company"] + 1)

        # Store clipping bounds
        self.clip_bounds["Monthly Income"] = X["Monthly Income"].quantile([0.01, 0.99])
        self.clip_bounds["Distance from Home"] = X["Distance from Home"].quantile([0.01, 0.99])
        # self.clip_bounds["Stagnation"] = stagnation.quantile([0.01, 0.99])
        # self.clip_bounds["Promotion_Rate"] = promotion_rate.quantile([0.01, 0.99])

        return self

    def transform(self, X):
        X = X.copy()

        """

        Feature Engineering did not improve performance, so we are skipping it for now.

        # --- Base features ---
        # X["Stagnation"] = X["Years at Company"] - X["Company Tenure"]
        X["Promotion_Rate"] = X["Number of Promotions"] / (X["Years at Company"] + 1)

        # --- Clip features using learned bounds ---
        for col in ["Monthly Income", "Distance from Home", "Promotion_Rate"]:
            q_low, q_high = self.clip_bounds[col]
            X[col] = X[col].clip(q_low, q_high)

        # --- Log transforms ---
        X["Monthly Income"] = np.log1p(X["Monthly Income"])
        X["Distance from Home"] = np.log1p(X["Distance from Home"])

        # --- Overtime Stress ---
        overtime_flag = (X["Overtime"].astype(str).str.lower() == "yes").astype(float)
        job_sat_map = {"Very Low":0.5, "Low":0.75, "Medium":1.0, "High":1.25, "Very High":1.5}
        job_sat = X["Job Satisfaction"].map(job_sat_map).fillna(1.0).astype(float)
        X["Overtime_Stress"] = overtime_flag * job_sat
        """

        # --- Clip features using learned bounds ---
        for col in ["Monthly Income", "Distance from Home"]:
            q_low, q_high = self.clip_bounds[col]
            X[col] = X[col].clip(q_low, q_high)

        # --- Drop unused ---
        drop_cols = [
            "Employee ID"
            # ,"Years at Company", "Age", "Company Tenure", "Number of Promotions", "Overtime", "Job Satisfaction",
        ]
        X = X.drop(columns=drop_cols, errors="ignore")


        return X

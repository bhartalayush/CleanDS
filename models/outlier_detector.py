"""
outlier_detector.py
Unsupervised Machine Learning for Anomaly & Outlier Detection.
Uses One-Class SVM with RBF kernel (with adaptive fallback to Isolation Forest for large N)
to identify multidimensional abnormal records in tabular data.
"""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class MLOutlierDetector:
    def __init__(self, contamination: float = 0.05):
        """
        contamination: Estimated proportion of outliers in the data (default 5%).
        """
        self.contamination = contamination

    def detect(self, df: pd.DataFrame) -> dict:
        """
        Detects multivariate outliers across numeric columns.
        Returns:
            - mask: Boolean Series (True = Outlier, False = Inlier)
            - count: Total number of outliers
            - percentage: Percentage of records flagged
            - algorithm: The ML algorithm used
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # If no numeric columns or too few rows, return empty mask
        if not numeric_cols or len(df) < 5:
            return {
                "mask": pd.Series(False, index=df.index),
                "count": 0,
                "percentage": 0.0,
                "algorithm": "None"
            }

        # Extract numeric matrix and fill temporary NaNs with median for distance estimation
        X = df[numeric_cols].copy()
        for col in numeric_cols:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0)

        # Scale features so variables with large scales do not dominate the kernel distance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_samples = len(df)

        # Adaptive Model Selection:
        # One-Class SVM has O(N^2) complexity; for datasets > 5000 rows, use IsolationForest
        if n_samples <= 5000:
            algo_name = "One-Class SVM (RBF Kernel)"
            model = OneClassSVM(nu=self.contamination, kernel="rbf", gamma="scale")
            preds = model.fit_predict(X_scaled)  # -1: Outlier, 1: Inlier
            is_outlier = (preds == -1)
        else:
            algo_name = "Isolation Forest"
            model = IsolationForest(contamination=self.contamination, random_state=42)
            preds = model.fit_predict(X_scaled)  # -1: Outlier, 1: Inlier
            is_outlier = (preds == -1)

        outlier_mask = pd.Series(is_outlier, index=df.index)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / n_samples * 100), 2)

        return {
            "mask": outlier_mask,
            "count": outlier_count,
            "percentage": outlier_pct,
            "algorithm": algo_name
        }

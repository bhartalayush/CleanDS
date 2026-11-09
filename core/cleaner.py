"""
cleaner.py
The central execution engine for CleanDS.
Coordinates column profiling, 1D-CNN categorization, sanitization,
deduplication, ML-driven imputation (MICE/k-NN), and outlier handling (One-Class SVM).
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from models.cnn_classifier import ColumnClassifier
from models.outlier_detector import MLOutlierDetector


class CleanDSEngine:
    def __init__(self):
        self.cnn_classifier = ColumnClassifier()
        self.outlier_detector = MLOutlierDetector()

    def clean(self, df: pd.DataFrame, policy: dict) -> tuple[pd.DataFrame, list[str]]:
        """
        Executes the cleaning pipeline on df guided by policy rules.
        Returns:
            (cleaned_df, audit_log)
        """
        df = df.copy()
        audit_log = []
        rules = policy.get("rules", {})
        initial_rows = len(df)

        # -------------------------------------------------------------
        # Step 1: Whitespace and Basic String Sanitization
        # -------------------------------------------------------------
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, 'null': np.nan, '': np.nan})

        audit_log.append("Sanitized text fields and normalized blank/null representations.")

        # -------------------------------------------------------------
        # Step 2: Deduplication
        # -------------------------------------------------------------
        if rules.get("drop_duplicates", True):
            dups_count = int(df.duplicated().sum())
            if dups_count > 0:
                df = df.drop_duplicates().reset_index(drop=True)
                audit_log.append(f"Removed {dups_count} exact duplicate record(s).")

        # -------------------------------------------------------------
        # Step 3: Column Role Classification (1D-CNN & Heuristics)
        # -------------------------------------------------------------
        detected_roles = {}
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                if series.nunique() == len(df) and any(k in str(col).lower() for k in ['id', 'key', 'uuid']):
                    detected_roles[col] = "Identifier"
                else:
                    detected_roles[col] = "Numeric"
            else:
                # Ambiguous string/object column -> Invoke pre-trained 1D-CNN
                samples = series.dropna().head(20).tolist()
                pred = self.cnn_classifier.predict_column(samples)
                detected_roles[col] = pred["role"]
                audit_log.append(f"1D-CNN classified '{col}' as {pred['role']} (Confidence: {pred['confidence']}%).")

        # -------------------------------------------------------------
        # Step 4: Formatting (Currencies & Dates)
        # -------------------------------------------------------------
        if rules.get("clean_currencies", True):
            for col, role in detected_roles.items():
                if role == "Currency" or "price" in str(col).lower() or "amount" in str(col).lower():
                    try:
                        cleaned_series = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                        df[col] = pd.to_numeric(cleaned_series, errors='coerce')
                        audit_log.append(f"Parsed and converted currency column '{col}' to numeric float.")
                    except Exception:
                        pass

        if rules.get("normalize_dates", True):
            for col, role in detected_roles.items():
                if role == "Date_Time" or "date" in str(col).lower() or "time" in str(col).lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        audit_log.append(f"Standardized date/time column '{col}' into ISO format.")
                    except Exception:
                        pass

        # -------------------------------------------------------------
        # Step 5: Handling Identifiers (Drop vs Keep based on Policy)
        # -------------------------------------------------------------
        id_action = rules.get("handle_identifiers", "keep")
        if id_action == "drop":
            cols_to_drop = [
                col for col, role in detected_roles.items()
                if role == "Identifier" or (df[col].nunique() == len(df) and any(k in str(col).lower() for k in ['id', 'uuid', 'key']))
            ]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                audit_log.append(f"Dropped identifier columns {cols_to_drop} to prevent model memorization.")

        # -------------------------------------------------------------
        # Step 6: Outlier Detection and Handling (One-Class SVM / Clip / Flag)
        # -------------------------------------------------------------
        outlier_strategy = rules.get("outlier_strategy", "clip")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols and outlier_strategy != "none":
            outlier_res = self.outlier_detector.detect(df)
            mask = outlier_res["mask"]
            count = outlier_res["count"]

            if count > 0:
                if outlier_strategy == "flag":
                    df["is_anomaly"] = mask.astype(int)
                    audit_log.append(f"Flagged {count} anomaly record(s) using {outlier_res['algorithm']} (new column 'is_anomaly').")
                elif outlier_strategy == "drop":
                    df = df[~mask].reset_index(drop=True)
                    audit_log.append(f"Removed {count} anomalous record(s) using {outlier_res['algorithm']}.")
                elif outlier_strategy == "clip":
                    for col in numeric_cols:
                        low = df[col].quantile(0.01)
                        high = df[col].quantile(0.99)
                        df[col] = df[col].clip(lower=low, upper=high)
                    audit_log.append("Clipped numeric features to [1st, 99th] percentiles to temper extreme outliers.")

        # -------------------------------------------------------------
        # Step 7: Intelligent Missing Value Imputation
        # -------------------------------------------------------------
        impute_strategy = rules.get("imputation_strategy", "mice")
        total_missing = int(df.isna().sum().sum())

        if total_missing > 0 and impute_strategy != "none":
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            # Categorical Imputation (Mode)
            for col in cat_cols:
                if df[col].isna().sum() > 0:
                    mode_val = df[col].mode(dropna=True)
                    fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
                    df[col] = df[col].fillna(fill_val)

            # Numeric Imputation (MICE / k-NN / Median)
            if num_cols and df[num_cols].isna().sum().sum() > 0:
                if impute_strategy == "mice":
                    try:
                        imputer = IterativeImputer(max_iter=10, random_state=42)
                        df[num_cols] = imputer.fit_transform(df[num_cols])
                        audit_log.append("Imputed missing numeric values using MICE (Iterative Multivariate Regression).")
                    except Exception:
                        for col in num_cols:
                            df[col] = df[col].fillna(df[col].median())
                        audit_log.append("Imputed numeric nulls via column median (MICE fallback).")

                elif impute_strategy == "knn":
                    try:
                        imputer = KNNImputer(n_neighbors=5)
                        df[num_cols] = imputer.fit_transform(df[num_cols])
                        audit_log.append("Imputed missing numeric values using k-Nearest Neighbors (k=5).")
                    except Exception:
                        for col in num_cols:
                            df[col] = df[col].fillna(df[col].median())
                        audit_log.append("Imputed numeric nulls via column median (k-NN fallback).")

                elif impute_strategy == "forward_fill":
                    df[num_cols] = df[num_cols].ffill().bfill()
                    audit_log.append("Applied forward-fill / backward-fill for temporal continuity.")

                elif impute_strategy == "median_mode":
                    for col in num_cols:
                        df[col] = df[col].fillna(df[col].median())
                    audit_log.append("Imputed missing numeric entries using feature median.")

        final_rows = len(df)
        audit_log.append(f"Pipeline complete: {initial_rows} initial rows -> {final_rows} cleaned rows.")

        return df, audit_log

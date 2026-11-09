"""
profiler.py
Inspects an input dataset to extract structural metrics, missingness statistics,
duplicate counts, and initial data type classifications.
"""

import pandas as pd

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Computes a comprehensive profile of the dataset:
    - Shape, memory usage
    - Duplicate row counts
    - Missing value percentages (dataset-wide & per column)
    - Initial heuristic column categorization
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())
    total_cells = total_rows * total_cols
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0

    columns_profile = []
    
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        
        # Fast heuristic classification (Tier 1)
        col_type = "Numeric"
        if pd.api.types.is_numeric_dtype(series):
            if unique_count == total_rows and ('id' in str(col).lower() or 'key' in str(col).lower()):
                col_type = "Identifier"
            elif unique_count == 2:
                col_type = "Boolean / Binary"
            else:
                col_type = "Numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_type = "Date / Time"
        elif pd.api.types.is_bool_dtype(series):
            col_type = "Boolean / Binary"
        else:
            # String / Object column -> Ambiguous (Candidate for Tier 2 CNN or regex checks)
            col_type = "Ambiguous_Text"

        sample_vals = series.dropna().astype(str).head(3).tolist()

        columns_profile.append({
            "name": col,
            "raw_dtype": str(series.dtype),
            "detected_type": col_type,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": unique_count,
            "samples": sample_vals
        })

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round((duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0.0,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "columns": columns_profile
    }

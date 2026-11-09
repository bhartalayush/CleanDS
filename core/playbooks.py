"""
playbooks.py
Defines the cleaning policies for domain-specific and general objectives.
Each playbook specifies how outliers, missing values, duplicates, and identifiers
should be handled according to the downstream task requirements.
"""

PLAYBOOKS = {
    # --- General Objectives (from UI screenshot) ---
    "Machine learning readiness": {
        "category": "General",
        "description": "Prepares dataset specifically for training ML models: removes unique identifiers to avoid leakage, imputes missing values via MICE, and clips extreme outliers.",
        "keywords": ["machine learning", "ml", "train", "model", "features", "classification", "regression", "target", "supervised", "predict"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "clip",          # Clip outliers to 1st and 99th percentiles
            "imputation_strategy": "mice",       # Use Iterative Random Forest / MICE
            "handle_identifiers": "drop",        # Drop IDs to prevent model overfitting
            "normalize_dates": True,
            "clean_currencies": True,
            "scale_numerics": False
        }
    },
    "Maximum data accuracy": {
        "category": "General",
        "description": "Strictest cleaning mode. Drops records with critical missing values, detects and removes corrupted outliers, and ensures high precision.",
        "keywords": ["accuracy", "high precision", "strict", "cleanest", "remove errors", "corrupted", "exact", "quality"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "drop",          # Remove rows with statistical anomalies
            "imputation_strategy": "mice",
            "handle_identifiers": "keep",
            "normalize_dates": True,
            "clean_currencies": True,
            "scale_numerics": False
        }
    },
    "Maximum data retention": {
        "category": "General",
        "description": "Prioritizes preserving every row. Never drops records: flags outliers instead of removing them, and aggressively imputes missing data.",
        "keywords": ["retention", "keep all rows", "do not drop", "preserve", "small dataset", "save data", "sparse"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "flag",          # Adds 'is_outlier' column, drops 0 rows
            "imputation_strategy": "median_mode",# Fast reliable imputation
            "handle_identifiers": "keep",
            "normalize_dates": True,
            "clean_currencies": True,
            "scale_numerics": False
        }
    },
    "Statistical analysis": {
        "category": "General",
        "description": "Optimized for hypothesis testing and distributions: avoids aggressive synthetic imputation that alters variance, caps extreme tails.",
        "keywords": ["statistics", "hypothesis testing", "p-value", "distribution", "variance", "anova", "biostatistics", "academic"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "flag",
            "imputation_strategy": "knn",        # Preserves local variance better than mean
            "handle_identifiers": "keep",
            "normalize_dates": True,
            "clean_currencies": True,
            "scale_numerics": False
        }
    },
    "Minimal intervention": {
        "category": "General",
        "description": "Performs only essential formatting and exact deduplication. Leaves numerical distributions and null values untouched.",
        "keywords": ["minimal", "raw", "as-is", "untouched", "simple", "only dedup", "formatting only", "no impute"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "none",
            "imputation_strategy": "none",
            "handle_identifiers": "keep",
            "normalize_dates": True,
            "clean_currencies": True,
            "scale_numerics": False
        }
    },
    "Anomaly / fraud detection": {
        "category": "General",
        "description": "Crucial for fraud analysis: NEVER deletes outliers because anomalies represent the target class. Flags anomalies and retains full audit trails.",
        "keywords": ["fraud", "anomaly", "suspicious", "scam", "theft", "unusual", "outlier detection", "intrusion", "cybersecurity"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "flag",          # Essential: keep fraud rows, flag with SVM
            "imputation_strategy": "median_mode",
            "handle_identifiers": "keep",        # Retain IDs for transaction tracing
            "normalize_dates": True,
            "clean_currencies": True,
            "scale_numerics": False
        }
    },

    # --- Domain Specific Objectives (from user's table) ---
    "Financial Accuracy": {
        "category": "Domain",
        "description": "Prioritize accurate numerical values, transaction records, dates and categorical entries while flagging suspicious data.",
        "keywords": ["finance", "banking", "transactions", "ledger", "accounting", "balance", "money", "payments", "invoices", "audit"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "flag",          # Suspicious transactions flagged via One-Class SVM
            "imputation_strategy": "knn",
            "handle_identifiers": "keep",        # Retain Account / Transaction IDs
            "normalize_dates": True,
            "clean_currencies": True,
            "scale_numerics": False
        }
    },
    "Clinical Data Integrity": {
        "category": "Domain",
        "description": "Ensure consistency and completeness while preserving rare or potentially significant observations.",
        "keywords": ["clinical", "healthcare", "medical", "patients", "hospital", "vitals", "diagnosis", "health", "doctor", "blood pressure"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "flag",          # Rare physiological observations must be preserved
            "imputation_strategy": "mice",       # Physiological correlations modeled via MICE
            "handle_identifiers": "keep",
            "normalize_dates": True,
            "clean_currencies": False,
            "scale_numerics": False
        }
    },
    "Sales Forecasting Readiness": {
        "category": "Domain",
        "description": "Identify missing periods, inconsistent categories, duplicate records and unusual demand fluctuations.",
        "keywords": ["sales", "forecasting", "revenue", "retail", "orders", "demand", "store", "e-commerce", "quarterly", "projections"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "clip",          # Dampen extreme one-off promotional spikes
            "imputation_strategy": "forward_fill",# Temporal continuity
            "handle_identifiers": "drop",
            "normalize_dates": True,
            "clean_currencies": True,
            "scale_numerics": False
        }
    },
    "Predictive Maintenance Readiness": {
        "category": "Domain",
        "description": "Detect faulty sensor readings, missing measurements and abnormal patterns while preserving genuine anomalies.",
        "keywords": ["maintenance", "sensors", "iot", "equipment", "machines", "factory", "breakdown", "hardware", "telemetry", "vibration", "temperature"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "flag",          # Machine failure signals are anomalies
            "imputation_strategy": "forward_fill",# Sensor continuity
            "handle_identifiers": "keep",        # Machine / Sensor IDs needed
            "normalize_dates": True,
            "clean_currencies": False,
            "scale_numerics": False
        }
    },
    "Customer Churn Prediction": {
        "category": "Domain",
        "description": "Prioritize reliable customer activity, subscription histories, billing records and usage patterns.",
        "keywords": ["churn", "retention", "subscribers", "customers", "attrition", "membership", "loyalty", "cancellation", "churn rate"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "clip",
            "imputation_strategy": "mice",
            "handle_identifiers": "drop",        # Drop customer IDs to avoid model overfitting
            "normalize_dates": True,
            "clean_currencies": True,
            "scale_numerics": False
        }
    },
    "Supply Chain Optimization": {
        "category": "Domain",
        "description": "Improve the consistency of inventory quantities, shipment records, delivery statuses and timestamps.",
        "keywords": ["supply chain", "logistics", "inventory", "warehouse", "shipping", "delivery", "freight", "suppliers", "stock", "tracking"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "clip",
            "imputation_strategy": "median_mode",
            "handle_identifiers": "keep",        # Tracking numbers and SKU IDs are essential
            "normalize_dates": True,
            "clean_currencies": True,
            "scale_numerics": False
        }
    },
    "Claims Data Integrity": {
        "category": "Domain",
        "description": "Identify duplicate records, inconsistent information, missing financial details and suspicious entries.",
        "keywords": ["insurance", "claims", "policy", "claimant", "settlement", "payout", "incident", "coverage"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "flag",          # Fraudulent claims flagged
            "imputation_strategy": "knn",
            "handle_identifiers": "keep",
            "normalize_dates": True,
            "clean_currencies": True,
            "scale_numerics": False
        }
    },
    "Campaign Analytics Readiness": {
        "category": "Domain",
        "description": "Prioritize accurate engagement metrics, conversion records, campaign labels and customer data.",
        "keywords": ["marketing", "campaign", "clicks", "impressions", "ads", "conversions", "ctr", "roas", "leads", "social media"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "clip",
            "imputation_strategy": "median_mode",
            "handle_identifiers": "drop",
            "normalize_dates": True,
            "clean_currencies": True,
            "scale_numerics": False
        }
    },
    "Workforce Analytics Readiness": {
        "category": "Domain",
        "description": "Standardize employee records, job titles, compensation details and employment histories.",
        "keywords": ["hr", "employees", "workforce", "salary", "payroll", "departments", "hiring", "human resources", "staff", "headcount"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "clip",
            "imputation_strategy": "median_mode",
            "handle_identifiers": "keep",
            "normalize_dates": True,
            "clean_currencies": True,
            "scale_numerics": False
        }
    },
    "Consumption Forecasting Readiness": {
        "category": "Domain",
        "description": "Identify missing readings, faulty measurements, inconsistent timestamps and unusual consumption patterns.",
        "keywords": ["energy", "electricity", "water", "utility", "consumption", "smart meter", "grid", "power", "kilowatt", "usage"],
        "rules": {
            "drop_duplicates": True,
            "outlier_strategy": "clip",
            "imputation_strategy": "forward_fill",
            "handle_identifiers": "keep",
            "normalize_dates": True,
            "clean_currencies": False,
            "scale_numerics": False
        }
    }
}

def get_playbook(objective_name: str) -> dict:
    """Retrieve playbook rules by exact objective name, falling back to ML readiness."""
    return PLAYBOOKS.get(objective_name, PLAYBOOKS["Machine learning readiness"])

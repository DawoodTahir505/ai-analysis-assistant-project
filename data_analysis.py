import pandas as pd

def get_basic_info(df):
    """Returns basic information about the dataset."""
    info = {
        "Number of rows": df.shape[0],
        "Number of columns": df.shape[1],
        "Column names": df.columns.tolist(),
        "Missing values": df.isnull().sum().to_dict(),
        "Duplicate rows": df.duplicated().sum(),
        "Data types": df.dtypes.astype(str).to_dict()
    }
    return info

def get_statistics(df):
    """Calculates useful statistics for numerical columns."""
    numerical_cols = df.select_dtypes(include=['number']).columns
    if len(numerical_cols) == 0:
        return None
    
    stats = {}
    for col in numerical_cols:
        stats[col] = {
            "Total (Sum)": float(df[col].sum()),
            "Average (Mean)": float(df[col].mean()),
            "Maximum": float(df[col].max()),
            "Minimum": float(df[col].min()),
            "Count": int(df[col].count())
        }
    return stats

def get_categorical_distribution(df):
    """Gets frequency distribution for categorical columns."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    distribution = {}
    for col in categorical_cols:
        # Keep top 10 categories to avoid huge outputs
        dist = df[col].value_counts().head(10).to_dict()
        distribution[col] = dist
    return distribution

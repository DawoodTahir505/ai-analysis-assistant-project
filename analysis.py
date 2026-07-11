# =============================================================================
# analysis.py
# Module for loading, analyzing, and answering questions about CSV datasets.
# Uses pandas and numpy for data processing.
# =============================================================================

import pandas as pd
import numpy as np


def load_dataset(file_path):
    """
    Load a CSV dataset and display basic information.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame or None: The loaded dataset, or None if loading fails.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        print("\n[+] Dataset loaded successfully!")
        print("\n" + "=" * 50)
        print("  DATASET INFORMATION")
        print("=" * 50)

        # Display the shape of the dataset (rows x columns)
        rows, columns = df.shape
        print(f"\n  Number of Rows    : {rows}")
        print(f"  Number of Columns : {columns}")

        # List all column names with numbering
        print(f"\n  Column Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"    {i}. {col}")

        # Show the data type of each column
        print(f"\n  Data Types:")
        for col in df.columns:
            print(f"    {col}: {df[col].dtype}")

        # Check for missing values in each column
        print(f"\n  Missing Values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("    No missing values found!")
        else:
            for col in df.columns:
                if missing[col] > 0:
                    print(f"    {col}: {missing[col]} missing")

        # Display the first 5 rows as a preview
        print(f"\n  First 5 Rows:")
        print(df.head().to_string(index=False))

        return df

    except FileNotFoundError:
        print(f"\n[!] Error: File '{file_path}' not found!")
        return None
    except pd.errors.EmptyDataError:
        print(f"\n[!] Error: File '{file_path}' is empty!")
        return None
    except Exception as e:
        print(f"\n[!] Error loading dataset: {e}")
        return None


def analyze_dataset(df):
    """
    Perform comprehensive statistical analysis on the dataset.
    Calculates statistics for both numeric and categorical columns.

    Parameters:
        df (pd.DataFrame): The dataset to analyze.

    Returns:
        dict: A dictionary containing all analysis results for AI explanation.
    """
    if df is None:
        print("[!] No dataset to analyze!")
        return None

    print("\n" + "=" * 50)
    print("  DATA ANALYSIS")
    print("=" * 50)

    # Store results for AI explanation later
    results = {}

    # Total number of records
    total_records = len(df)
    print(f"\n  Total Records: {total_records}")
    results['total_records'] = total_records

    # --- Numeric Column Statistics ---
    print(f"\n  Numeric Column Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results['numeric_stats'] = {}

    if len(numeric_cols) > 0:
        for col in numeric_cols:
            stats = {
                'average': round(df[col].mean(), 2),
                'maximum': df[col].max(),
                'minimum': df[col].min(),
                'total': round(df[col].sum(), 2),
                'median': round(df[col].median(), 2),
                'std_dev': round(df[col].std(), 2),
                'count': df[col].count()
            }
            results['numeric_stats'][col] = stats

            print(f"\n    Column: {col}")
            print(f"      Count   : {stats['count']}")
            print(f"      Average : {stats['average']}")
            print(f"      Maximum : {stats['maximum']}")
            print(f"      Minimum : {stats['minimum']}")
            print(f"      Total   : {stats['total']}")
            print(f"      Median  : {stats['median']}")
            print(f"      Std Dev : {stats['std_dev']}")
    else:
        print("    No numeric columns found!")

    # --- Categorical Column Analysis ---
    print(f"\n  Categorical Column Analysis:")
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    results['categorical_stats'] = {}

    if len(categorical_cols) > 0:
        for col in categorical_cols:
            frequency = df[col].value_counts()
            mode_series = df[col].mode()
            cat_stats = {
                'unique_values': df[col].nunique(),
                'most_common': mode_series[0] if not mode_series.empty else "N/A",
                'distribution': {}
            }

            print(f"\n    Column: {col}")
            print(f"      Unique Values : {cat_stats['unique_values']}")
            print(f"      Most Common   : {cat_stats['most_common']}")
            print(f"      Frequency Distribution:")

            for value, count in frequency.items():
                percentage = round((count / len(df)) * 100, 1)
                cat_stats['distribution'][value] = {
                    'count': int(count),
                    'percentage': percentage
                }
                print(f"        {value}: {count} ({percentage}%)")

            results['categorical_stats'][col] = cat_stats
    else:
        print("    No categorical columns found!")

    # --- Overall Summary ---
    print(f"\n  Overall Summary:")
    print(f"    Shape          : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"    Missing Values : {df.isnull().sum().sum()} total")
    results['shape'] = df.shape
    results['total_missing'] = int(df.isnull().sum().sum())

    return results


def answer_question(df, question):
    """
    Answer a natural language question about the dataset using Python logic.
    Supports various question patterns by analyzing keywords and matching
    them to appropriate pandas operations.

    Parameters:
        df (pd.DataFrame): The dataset to query.
        question (str): The natural language question to answer.

    Returns:
        str: The answer to the question.
    """
    if df is None:
        return "No dataset loaded to answer questions."

    # Normalize the question for easier keyword matching
    q = question.lower().strip()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    all_cols = df.columns.tolist()

    # --- Find which column the question is about ---
    target_col = None
    for col in all_cols:
        if col.lower() in q:
            target_col = col
            break

    # --- Pattern: "average" or "mean" ---
    if any(word in q for word in ['average', 'mean']):
        if target_col and target_col in numeric_cols:
            avg = round(df[target_col].mean(), 2)
            return f"The average {target_col} is {avg}."
        # Try to find any numeric column mentioned
        for col in numeric_cols:
            if col.lower() in q:
                avg = round(df[col].mean(), 2)
                return f"The average {col} is {avg}."
        # If no specific column, show all averages
        return "The average of all numeric columns:\n" + "\n".join(
            [f"  - {col}: {round(df[col].mean(), 2)}" for col in numeric_cols]
        )

    # --- Pattern: "maximum" or "highest" or "max" or "most" ---
    if any(word in q for word in ['maximum', 'highest', 'max', 'most', 'top', 'best']):
        if target_col and target_col in numeric_cols:
            max_val = df[target_col].max()
            return f"The maximum {target_col} is {max_val}."
        for col in numeric_cols:
            if col.lower() in q:
                max_val = df[col].max()
                return f"The maximum {col} is {max_val}."

    # --- Pattern: "minimum" or "lowest" or "min" ---
    if any(word in q for word in ['minimum', 'lowest', 'min', 'least', 'worst']):
        if target_col and target_col in numeric_cols:
            min_val = df[target_col].min()
            return f"The minimum {target_col} is {min_val}."
        for col in numeric_cols:
            if col.lower() in q:
                min_val = df[col].min()
                return f"The minimum {col} is {min_val}."

    # --- Pattern: "how many" or "total" or "count" ---
    if any(word in q for word in ['how many', 'total', 'count', 'number of']):
        if 'rows' in q or 'records' in q or 'students' in q or 'entries' in q:
            return f"There are {len(df)} total records in the dataset."
        if target_col and target_col in categorical_cols:
            counts = df[target_col].value_counts()
            result = f"Counts for {target_col}:\n"
            for val, cnt in counts.items():
                result += f"  - {val}: {cnt}\n"
            return result.strip()

    # --- Pattern: "percentage" or "percent" or "%" ---
    if any(word in q for word in ['percentage', 'percent', '%']):
        if target_col and target_col in categorical_cols:
            counts = df[target_col].value_counts(normalize=True) * 100
            result = f"Percentage distribution for {target_col}:\n"
            for val, pct in counts.items():
                result += f"  - {val}: {pct:.1f}%\n"
            return result.strip()

    # --- Pattern: "most common" or "most frequent" or "frequently" ---
    if any(word in q for word in ['most common', 'most frequent', 'frequently', 'popular']):
        if target_col and target_col in categorical_cols:
            mode_series = df[target_col].mode()
            if not mode_series.empty:
                mode = mode_series[0]
                count = df[target_col].value_counts()[mode]
                return f"The most common {target_col} is '{mode}' with {count} occurrences."
            return f"No common value found for {target_col}."

    # --- Pattern: "correlation" or "relationship" ---
    if 'correlation' in q or 'relationship' in q:
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            return f"Correlation matrix:\n{corr.to_string()}"

    # --- Pattern: "distribution" ---
    if 'distribution' in q:
        if target_col:
            if target_col in categorical_cols:
                dist = df[target_col].value_counts()
                result = f"Distribution of {target_col}:\n"
                for val, cnt in dist.items():
                    result += f"  - {val}: {cnt}\n"
                return result.strip()
            elif target_col in numeric_cols:
                desc = df[target_col].describe()
                return f"Distribution of {target_col}:\n{desc.to_string()}"

    # --- Fallback: try to provide a general answer ---
    if target_col:
        if target_col in numeric_cols:
            stats = df[target_col].describe()
            return f"Statistics for {target_col}:\n{stats.to_string()}"
        elif target_col in categorical_cols:
            counts = df[target_col].value_counts()
            return f"Value counts for {target_col}:\n{counts.to_string()}"

    return (
        "I could not determine the exact answer. "
        "Please try rephrasing with a specific column name and metric "
        f"(e.g., average, max, min).\nAvailable columns: {', '.join(all_cols)}"
    )

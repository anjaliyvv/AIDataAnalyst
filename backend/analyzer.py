import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

def load_dataset(file_path):
    """
    Loads a CSV or Excel dataset.
    """

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)

    else:
        raise ValueError("Only CSV and XLSX files are supported.")

    return df


# --------------------------------------------------
# 2. DATASET INFORMATION
# --------------------------------------------------

def get_dataset_info(df):
    """
    Returns basic information about the dataset.
    """

    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict()
    }

    return info


# --------------------------------------------------
# 3. MISSING VALUES
# --------------------------------------------------

def get_missing_values(df):
    """
    Counts missing values in every column.
    """

    return df.isnull().sum()


# --------------------------------------------------
# 4. DUPLICATES
# --------------------------------------------------

def get_duplicates(df):
    """
    Counts duplicate rows.
    """

    return df.duplicated().sum()


# --------------------------------------------------
# 5. DESCRIPTIVE STATISTICS
# --------------------------------------------------

def get_statistics(df):
    """
    Generates descriptive statistics
    for numerical columns.
    """

    return df.describe()


# --------------------------------------------------
# 6. CORRELATION
# --------------------------------------------------

def get_correlations(df):
    """
    Calculates correlations between numerical columns.
    """

    numeric_df = df.select_dtypes(include=np.number)

    return numeric_df.corr()


# --------------------------------------------------
# 7. UNIQUE VALUES
# --------------------------------------------------

def get_unique_values(df):
    """
    Returns number of unique values
    present in every column.
    """

    return df.nunique()


# --------------------------------------------------
# 8. OUTLIER DETECTION
# --------------------------------------------------

def detect_outliers(df):
    """
    Detects outliers in numerical columns
    using the IQR method.
    """

    outlier_results = {}

    numeric_columns = df.select_dtypes(include=np.number).columns

    for column in numeric_columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_limit = Q1 - (1.5 * IQR)
        upper_limit = Q3 + (1.5 * IQR)

        outliers = df[
            (df[column] < lower_limit)
            | (df[column] > upper_limit)
        ]

        outlier_results[column] = len(outliers)

    return outlier_results


# --------------------------------------------------
# 9. CATEGORY SUMMARY
# --------------------------------------------------

def category_summary(df, column):
    """
    Counts the occurrences of each value
    in a categorical column.
    """

    if column not in df.columns:
        return "Column not found."

    return df[column].value_counts()


# --------------------------------------------------
# 10. GROUP ANALYSIS
# --------------------------------------------------

def group_analysis(df, group_column, value_column):
    """
    Groups data by a categorical column and
    calculates the sum of a numerical column.
    """

    if group_column not in df.columns:
        return "Group column not found."

    if value_column not in df.columns:
        return "Value column not found."

    return (
        df.groupby(group_column)[value_column]
        .sum()
        .sort_values(ascending=False)
    )


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    print("\n================================")
    print("       AI DATA ANALYST")
    print("================================")

    # Load our test dataset
    df = load_dataset("datasets/sales.csv")

    # ----------------------------------------------
    # Preview
    # ----------------------------------------------

    print("\n--- DATASET PREVIEW ---")
    print(df.head())

    # ----------------------------------------------
    # Basic information
    # ----------------------------------------------

    print("\n--- DATASET INFORMATION ---")

    info = get_dataset_info(df)

    print("Rows:", info["rows"])
    print("Columns:", info["columns"])
    print("Column Names:", info["column_names"])

    print("\nData Types:")

    for column, dtype in info["data_types"].items():
        print(f"{column}: {dtype}")

    # ----------------------------------------------
    # Missing values
    # ----------------------------------------------

    print("\n--- MISSING VALUES ---")
    print(get_missing_values(df))

    # ----------------------------------------------
    # Duplicate rows
    # ----------------------------------------------

    print("\n--- DUPLICATE ROWS ---")
    print(get_duplicates(df))

    # ----------------------------------------------
    # Statistics
    # ----------------------------------------------

    print("\n--- STATISTICS ---")
    print(get_statistics(df))

    # ----------------------------------------------
    # Unique values
    # ----------------------------------------------

    print("\n--- UNIQUE VALUES ---")
    print(get_unique_values(df))

    # ----------------------------------------------
    # Correlation
    # ----------------------------------------------

    print("\n--- CORRELATION MATRIX ---")
    print(get_correlations(df))

    # ----------------------------------------------
    # Outliers
    # ----------------------------------------------

    print("\n--- OUTLIERS ---")

    outliers = detect_outliers(df)

    for column, count in outliers.items():
        print(f"{column}: {count} outlier(s)")

    # ----------------------------------------------
    # Category analysis
    # ----------------------------------------------

    print("\n--- SALES BY CATEGORY ---")

    print(
        group_analysis(
            df,
            "Category",
            "Revenue"
        )
    )

    # ----------------------------------------------
    # Region analysis
    # ----------------------------------------------

    print("\n--- SALES BY REGION ---")

    print(
        group_analysis(
            df,
            "Region",
            "Revenue"
        )
    )

    print("\n================================")
    print("Analysis completed successfully!")
    print("================================")
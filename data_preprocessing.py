import pandas as pd


COLUMN_ALIASES = {
    "CustomerID": "Customer_ID",
    "Customer Name": "Customer_Name",
    "CustomerName": "Customer_Name",
    "Annual Income (k$)": "Annual_Income_INR",
    "Spending Score (1-100)": "Spending_Score",
}


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    normalized = df.copy()
    normalized.columns = [column.strip() for column in normalized.columns]
    normalized.rename(columns=COLUMN_ALIASES, inplace=True)

    required_columns = [
        "Customer_ID",
        "Customer_Name",
        "Gender",
        "Age",
        "Annual_Income_INR",
        "Spending_Score",
    ]

    missing_columns = [column for column in required_columns if column not in normalized.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    normalized = normalized[required_columns].copy()

    numeric_columns = ["Customer_ID", "Age", "Annual_Income_INR", "Spending_Score"]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(
            normalized[column].astype(str).str.replace(r"[₹,]", "", regex=True),
            errors="coerce",
        )

    normalized["Customer_Name"] = normalized["Customer_Name"].astype(str).str.strip()
    normalized["Gender"] = normalized["Gender"].astype(str).str.strip().str.title()

    normalized.dropna(inplace=True)
    normalized.drop_duplicates(inplace=True)

    normalized["Customer_ID"] = normalized["Customer_ID"].astype(int)
    normalized["Age"] = normalized["Age"].astype(int)
    normalized["Annual_Income_INR"] = normalized["Annual_Income_INR"].astype(int)
    normalized["Spending_Score"] = normalized["Spending_Score"].astype(int)

    return normalized

def profile_customers(df):
    return (
        df.groupby("Cluster")
        .agg(
            {
                "Age": "mean",
                "Annual_Income_INR": "mean",
                "Spending_Score": "mean",
                "Frequency": "mean",
                "Monetary": "mean",
            }
        )
        .round(2)
    )
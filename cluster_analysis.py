def analyze_clusters(df):

    print("\nCluster Statistics\n")

    summary = df.groupby('Cluster').agg({
        'Annual Income (k$)': ['mean','min','max'],
        'Spending Score (1-100)': ['mean','min','max']
    })

    print(summary)

    print("\nCustomer Count Per Cluster\n")
    print(df['Cluster'].value_counts())
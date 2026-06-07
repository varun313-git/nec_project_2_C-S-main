from pathlib import Path

from src.clustering import perform_clustering, scale_features
from src.customer_profiles import profile_customers
from src.data_preprocessing import load_data, preprocess_data
from src.pca_analysis import apply_pca
from src.rfm_analysis import create_rfm
from src.visualization import plot_clusters, plot_pca_projection


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "mall_customers_india.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	data = load_data(DATA_PATH)
	data = preprocess_data(data)
	data = create_rfm(data)

	features = data[
		[
			"Age",
			"Annual_Income_INR",
			"Spending_Score",
			"Recency",
			"Frequency",
			"Monetary",
		]
	]

	scaled_features = scale_features(features)
	pca_data, pca_model = apply_pca(scaled_features)
	labels, _ = perform_clustering(scaled_features)

	data = data.copy()
	data["Cluster"] = labels

	profile = profile_customers(data)

	data.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
	profile.to_csv(OUTPUT_DIR / "segment_summary.csv")

	plot_clusters(pca_data, labels, OUTPUT_DIR / "cluster_plot.png")
	plot_pca_projection(pca_data, labels, pca_model, OUTPUT_DIR / "pca_plot.png")

	print(profile)
	return data, profile


if __name__ == "__main__":
	main()

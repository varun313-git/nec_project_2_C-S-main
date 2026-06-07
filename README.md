# Market & Customer Segmentation Analysis

This project segments mall customers using age, annual income, spending score, and synthetic RFM features. It standardizes the data, reduces the feature space with PCA, clusters customers with K-Means, and exports both tabular and visual outputs.

## Project Structure

- `data/mall_customers_india.csv` - modified sample dataset with INR income values
- `notebooks/customer_segmentation.ipynb` - notebook walkthrough of the pipeline
- `src/` - preprocessing, RFM, PCA, clustering, visualization, and profiling helpers
- `outputs/` - generated CSV files and plots

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The script generates:

- `outputs/customer_segments.csv`
- `outputs/segment_summary.csv`
- `outputs/cluster_plot.png`
- `outputs/pca_plot.png`

## Notes

The dataset uses `Annual_Income_INR` values for visualization and analysis. Synthetic RFM fields are generated because the mall dataset does not contain transaction history.

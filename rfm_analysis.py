import numpy as np


def create_rfm(df, random_state=42):
	rng = np.random.default_rng(random_state)
	rfm_frame = df.copy()

	rfm_frame["Recency"] = rng.integers(1, 100, len(rfm_frame))
	rfm_frame["Frequency"] = rng.integers(1, 20, len(rfm_frame))
	rfm_frame["Monetary"] = rfm_frame["Annual_Income_INR"]

	return rfm_frame

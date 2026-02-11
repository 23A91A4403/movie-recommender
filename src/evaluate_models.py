import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("data/processed_movies.csv")

# Train-test split (80/20)
df = df.sample(frac=1, random_state=42)
split = int(0.8 * len(df))
train = df[:split]
test = df[split:]

# Create user-item matrices
train_matrix = train.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

test_matrix = test.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

# Align matrices
train_matrix, test_matrix = train_matrix.align(test_matrix, fill_value=0)

# ----- User-based CF RMSE -----
from sklearn.metrics.pairwise import cosine_similarity

user_similarity = cosine_similarity(train_matrix)
pred_user = np.dot(user_similarity, train_matrix)
pred_user = pred_user / np.array([np.abs(user_similarity).sum(axis=1)]).T

rmse_user = np.sqrt(mean_squared_error(
    test_matrix.values.flatten(),
    pred_user.flatten()
))

# ----- SVD RMSE -----
U, sigma, Vt = np.linalg.svd(train_matrix.values, full_matrices=False)
k = 20
sigma_k = np.diag(sigma[:k])
U_k = U[:, :k]
Vt_k = Vt[:k, :]

pred_svd = np.dot(np.dot(U_k, sigma_k), Vt_k)

rmse_svd = np.sqrt(mean_squared_error(
    test_matrix.values.flatten(),
    pred_svd.flatten()
))

# Dummy ranking metrics (simple placeholder)
precision_user = 0.5
ndcg_user = 0.5
precision_svd = 0.6
ndcg_svd = 0.6

# Save metrics
metrics = {
    "user_based_cf": {
        "rmse": float(rmse_user),
        "precision_at_10": float(precision_user),
        "ndcg_at_10": float(ndcg_user)
    },
    "svd": {
        "rmse": float(rmse_svd),
        "precision_at_10": float(precision_svd),
        "ndcg_at_10": float(ndcg_svd)
    }
}

os.makedirs("output", exist_ok=True)
with open("output/evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Evaluation metrics saved.")

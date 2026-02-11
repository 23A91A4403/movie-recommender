import pandas as pd
import numpy as np
import os

# Load data
df = pd.read_csv("data/processed_movies.csv")

# Create user-item matrix
user_item_matrix = df.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

# Convert to numpy array
matrix = user_item_matrix.values

# Apply SVD
U, sigma, Vt = np.linalg.svd(matrix, full_matrices=False)

# Reduce dimensions (latent features)
k = 20
sigma_k = np.diag(sigma[:k])
U_k = U[:, :k]
Vt_k = Vt[:k, :]

# Reconstruct predicted ratings
predicted_ratings = np.dot(np.dot(U_k, sigma_k), Vt_k)

pred_df = pd.DataFrame(
    predicted_ratings,
    index=user_item_matrix.index,
    columns=user_item_matrix.columns
)

# Get recommendations for user 1
user_id = 1
user_ratings = user_item_matrix.loc[user_id]
predicted_user = pred_df.loc[user_id]

# Remove already rated movies
unrated = user_ratings[user_ratings == 0].index
recommendations = predicted_user[unrated]

# Get top 10
top_10 = recommendations.sort_values(ascending=False).head(10)

# Get movie titles
movie_titles = df[["movie_id", "title"]].drop_duplicates()
results = []

for movie_id, rating in top_10.items():
    title = movie_titles[movie_titles["movie_id"] == movie_id]["title"].values[0]
    results.append([movie_id, title, rating])

# Save output
os.makedirs("output", exist_ok=True)
output_df = pd.DataFrame(results, columns=["movie_id", "title", "estimated_rating"])
output_df.to_csv("output/svd_recommendations.csv", index=False)

print("SVD recommendations saved.")

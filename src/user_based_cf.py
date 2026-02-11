import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load processed data
df = pd.read_csv("data/processed_movies.csv")

# Create user-item matrix
user_item_matrix = df.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

# Compute user similarity
user_similarity = cosine_similarity(user_item_matrix)

# Convert to DataFrame
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

# Get recommendations for user 1
target_user = 1

# Find similar users
similar_users = user_similarity_df[target_user].sort_values(ascending=False)[1:6]

# Weighted ratings
weighted_ratings = pd.Series(0, index=user_item_matrix.columns)

for sim_user, similarity in similar_users.items():
    weighted_ratings += similarity * user_item_matrix.loc[sim_user]

# Remove movies already rated by user
rated_movies = user_item_matrix.loc[target_user]
weighted_ratings = weighted_ratings[rated_movies == 0]

# Get top 10 movies
top_movies = weighted_ratings.sort_values(ascending=False).head(10)

# Get movie titles
movie_titles = df[["movie_id", "title"]].drop_duplicates()
results = []

for movie_id, score in top_movies.items():
    title = movie_titles[movie_titles["movie_id"] == movie_id]["title"].values[0]
    results.append([movie_id, title, score])

# Save results
os.makedirs("output", exist_ok=True)
output_df = pd.DataFrame(results, columns=["movie_id", "title", "estimated_rating"])
output_df.to_csv("output/user_based_recommendations.csv", index=False)

print("User-based recommendations saved.")

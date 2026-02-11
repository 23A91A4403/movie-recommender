import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("data/processed_movies.csv")

# Get unique movies
movies = df[["movie_id", "title", "genres"]].drop_duplicates()

# Convert genres to TF-IDF vectors
tfidf = TfidfVectorizer(tokenizer=lambda x: x.split("|"))
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Compute similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=movies["movie_id"],
    columns=movies["movie_id"]
)

# Get user 1's watched movies
user_id = 1
user_movies = df[df["user_id"] == user_id]["movie_id"].unique()

# Score movies based on similarity
scores = pd.Series(0, index=movies["movie_id"])

for movie in user_movies:
    if movie in similarity_df:
        scores += similarity_df[movie]

# Remove already watched movies
scores = scores[~scores.index.isin(user_movies)]

# Get top 10
top_10 = scores.sort_values(ascending=False).head(10)

# Prepare results
results = []
for movie_id, score in top_10.items():
    title = movies[movies["movie_id"] == movie_id]["title"].values[0]
    results.append([movie_id, title, score])

# Save output
os.makedirs("output", exist_ok=True)
output_df = pd.DataFrame(results, columns=["movie_id", "title", "similarity_score"])
output_df.to_csv("output/content_based_recommendations.csv", index=False)

print("Content-based recommendations saved.")

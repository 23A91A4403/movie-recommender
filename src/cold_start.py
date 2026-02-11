import pandas as pd
import os

# Load data
df = pd.read_csv("data/processed_movies.csv")

# Calculate average rating per movie
avg_ratings = df.groupby(["movie_id", "title"])["rating"].mean().reset_index()

# Rename column
avg_ratings.rename(columns={"rating": "average_rating"}, inplace=True)

# Sort by rating
top_10 = avg_ratings.sort_values(by="average_rating", ascending=False).head(10)

# Save output
os.makedirs("output", exist_ok=True)
top_10.to_csv("output/cold_start_recommendations.csv", index=False)

print("Cold-start recommendations saved.")

import pandas as pd
import os

# File paths
data_path = "data/u.data"
item_path = "data/u.item"
output_path = "data/processed_movies.csv"

# Load ratings data
ratings = pd.read_csv(
    data_path,
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
    encoding="latin-1"
)

# Load movie metadata
items = pd.read_csv(
    item_path,
    sep="|",
    encoding="latin-1",
    header=None
)

# Movie columns (based on MovieLens 100k format)
genre_columns = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]

item_columns = ["movie_id", "title", "release_date", "video_release_date",
                "IMDb_URL"] + genre_columns

items.columns = item_columns

# Create genres column as pipe-separated string
def get_genres(row):
    genres = [genre for genre in genre_columns if row[genre] == 1]
    return "|".join(genres)

items["genres"] = items.apply(get_genres, axis=1)

# Keep only needed columns
items = items[["movie_id", "title", "genres"]]

# Merge ratings and movie data
merged = pd.merge(ratings, items, on="movie_id")

# Keep required columns
processed = merged[["user_id", "movie_id", "rating", "title", "genres"]]

# Save processed file
os.makedirs("data", exist_ok=True)
processed.to_csv(output_path, index=False)

print("Processed data saved to:", output_path)

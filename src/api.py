from fastapi import FastAPI
import pandas as pd

app = FastAPI()

# Load recommendation files
svd_file = "output/svd_recommendations.csv"
cold_start_file = "output/cold_start_recommendations.csv"

svd_df = pd.read_csv(svd_file)
cold_df = pd.read_csv(cold_start_file)

# Load user data to check existence
data = pd.read_csv("data/processed_movies.csv")
existing_users = set(data["user_id"].unique())

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int):
    if user_id in existing_users:
        recs = svd_df.to_dict(orient="records")
        return {
            "user_id": user_id,
            "recommendations": recs
        }
    else:
        recs = cold_df.rename(
            columns={"average_rating": "estimated_rating"}
        ).to_dict(orient="records")
        return {
            "user_id": user_id,
            "recommendations": recs
        }

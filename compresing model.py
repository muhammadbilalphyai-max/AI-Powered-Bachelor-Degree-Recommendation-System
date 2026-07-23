import joblib
import os

model = joblib.load("bachelor_degree_recommendation_model.joblib")

joblib.dump(
    model,
    "bachelor_degree_recommendation_model_compressed.joblib",
    compress=3
)

print("Original:",
      os.path.getsize("bachelor_degree_recommendation_model.joblib") / (1024*1024), "MB")

print("Compressed:",
      os.path.getsize("bachelor_degree_recommendation_model_compressed.joblib") / (1024*1024), "MB")
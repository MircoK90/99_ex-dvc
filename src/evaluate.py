import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

def main():
    X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv")

    with open("models/gbr_model.pkl", "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_test)

    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(preds, columns=["prediction"]).to_csv("data/processed_data/prediction.csv", index=False)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump({"mse": mse, "r2": r2}, f)

if __name__ == "__main__":
    main()

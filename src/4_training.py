import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

def main():
    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")

    with open("models/best_params.pkl", "rb") as f:
        params = pickle.load(f)

    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train.values.ravel())

    with open("models/gbr_model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
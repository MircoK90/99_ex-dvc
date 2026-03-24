import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import os

def main():
    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")

    model = GradientBoostingRegressor()

    params = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4]
    }

    grid = GridSearchCV(model, params, cv=3, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train.values.ravel())

    os.makedirs("models", exist_ok=True)

    with open("models/best_params.pkl", "wb") as f:
        pickle.dump(grid.best_params_, f)

if __name__ == "__main__":
    main()

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    model = joblib.load("models/model.joblib")
    df = pd.read_csv("data/iris.csv")
    X = df.drop(columns=["species"])
    y = df["species"]
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.85, f"Model accuracy too low: {acc}"
    print(f"Model accuracy: {acc}")

if __name__ == "__main__":
    test_model_accuracy()
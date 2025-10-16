import pandas as pd

def test_data_shape():
    df = pd.read_csv("data/iris.csv")
    assert df.shape[1] == 5, "Expected 5 columns in dataset"

def test_no_missing_values():
    df = pd.read_csv("data/iris.csv")
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"

import pandas as pd
from src.preprocess import load_and_preprocess

def test_preprocess():
    df = load_and_preprocess("data/train.csv")
    assert df["Age"].isna().sum() == 0, "Age should have no NaN after imputation"
    assert "Survived" in df.columns, "Survived column missing"
    assert df["Sex"].dtype == "int64", "Sex should be encoded as 0/1"
    print("✅ All preprocessing tests passed")

import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df = df[['Survived', 'Pclass', 'Sex', 'Age']]
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())

    return df
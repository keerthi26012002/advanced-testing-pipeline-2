import pandas as pd
from sklearn.datasets import load_iris


def get_iris_dataframe():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df


def test_dataset_has_expected_columns():
    df = get_iris_dataframe()

    expected_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "target"
    ]

    assert list(df.columns) == expected_columns, "Dataset schema mismatch"


def test_dataset_has_no_nulls():
    df = get_iris_dataframe()
    assert df.isnull().sum().sum() == 0, "Dataset contains null values"


def test_dataset_row_count_positive():
    df = get_iris_dataframe()
    assert len(df) > 0, "Dataset is empty"
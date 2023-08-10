import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def get_sample_size(x):
    counts = len(x)
    return counts * 2 if counts < 1000 else counts

def rebalance_data(df):
    df = (
        df.groupby("steering")
        .apply(lambda x: x.sample(n=get_sample_size(x), replace=True))
        .reset_index(drop=True)
    )
    return df


def load_data(data_dir, test_size):
    """
    load simulator driving training data
    """
    df = pd.read_csv(
        os.path.join(os.getcwd(), data_dir, "driving_log.csv"),
        names=["center", "left", "right", "steering", "throttle", "reverse", "speed"],
    )
    df = rebalance_data(df)
    X = df[["center", "left", "right"]].values
    Y = df["steering"].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    X_train, Y_train = shuffle(X_train, Y_train)
    X_test, Y_test = shuffle(X_test, Y_test)
    return X_train, X_test, Y_train, Y_test

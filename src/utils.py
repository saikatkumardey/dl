import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def sigmoid(Z: np.array) -> np.array:
    """
    computes sigmoid()

    Parameters
    ---------
    Z: a numpy array

    Returns
    ---------
    numpy array
    """
    return 1 / (1 + np.exp(-Z))


def sigmoid_prime(A: np.array) -> np.array:
    """
    Computes the derivative of sigmoid function

    Parameters
    ----------
    A: an numpy array of sigmoid()

    Returns
    --------
    Derivative of the sigmoid() over the entire array
    """
    return A * (1 - A)


def compute_cost(Y: np.array, Y_hat: np.array) -> float:
    """
    Computes the cost function - 1/m*[y*log(y_hat) + (1-y)*log(1-y_hat)]

    Parameters
    ----------
    Y: actual labels
    Y_hat: predicted labels

    Returns
    ---------
    a float
    """
    return -np.mean([Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)])


def load_data(path: str = "../data/haberman.data") -> [np.ndarray, np.ndarray]:
    """
    Loads data, specifically https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival

    Parameters
    ----------
    Path to training data

    Returns
    ---------
    a tuple (X,Y) 
    where X = [n,m] matrix of n features, m training samples
    Y = [1,m] vector of m labels
    """

    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df = pd.read_csv(path, header=None)
    df[3] = df[3].map({2: 0, 1: 1})  # 1 = survived, 2 = died

    X = scaler.fit_transform(df[[0, 1, 2]].values).T
    Y = df[3].values.reshape(1, -1)

    return X, Y


def baseline(X, Y) -> float:
    """
    Baseline accuracy from standard library for comparison.
    Note that the shape of X/Y are transpose of what has been used for our implementation.

    Parameters
    ----------

    X: [m,n] training matrix
    Y: [m,1] output matrix

    Returns
    --------
    An accuracy score (0-100%)
    """

    lr = LogisticRegression()
    lr.fit(X, Y)
    y_pred = lr.predict(X)
    return np.round(accuracy_score(y_true=np.ravel(Y), y_pred=np.ravel(y_pred)), 3) * 100

import numpy as np

np.random.seed(1)


# Define parameters
def init_parameters(n_x: int, n_h: int, n_y: int = 1) -> dict:
    """
    n_x : number of features

    returns: a dict of parameters W & b
    """
    return {
        "W1": np.random.randn(n_h, n_x) * 0.01,  # one weight per feature,
        "W2": np.random.randn(n_y, n_h) * 0.01,
        "b1": np.zeros((n_h, 1)),  # one bias
        "b2": np.zeros((n_y, 1)),
    }


# Define activation function
def sigmoid(Z: np.array) -> np.array:
    """
    computes sigmoid of Z
    """
    return 1 / (1 + np.exp(-Z))


def sigmoid_backward(A: np.array) -> np.array:
    """
    A is sigmoid()
    """
    return A * (1 - A)


# Compute Cost
def compute_cost(y: np.array, y_hat: np.array) -> float:
    """
    y: actual
    y_hat: predictions
    """
    return -np.mean([y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)])


# Forward propagation
def forward_prop(X: np.ndarray, params: dict) -> [np.ndarray, dict]:
    W1 = params["W1"]  # [n_h,n_x]
    b1 = params["b1"]  # [n_h,1]

    W2 = params["W2"]  # [n_y,n_h]
    b2 = params["b2"]  # [n_y,1]

    Z1 = np.dot(W1, X) + b1  # [n_h,m] = [n_h,n_x] * [n_x,m] + [1,m]
    A1 = sigmoid(Z1)  # [n_h,m]
    Z2 = np.dot(W2, A1) + b2  # [n_y,m] = [n_y,n_h] * [n_h,m] + [n_y,1]
    A2 = sigmoid(Z2)  # [n_y,m]

    cache = dict(A1=A1, A2=A2)

    return A2, cache


# Backward propagation


# Compute gradients
def compute_grads(X: np.ndarray, Y: np.ndarray, cache: dict, params: dict) -> dict:

    m = X.shape[0]
    A1 = cache["A1"]  # [n_h,m]
    A2 = cache["A2"]  # [n_y,m]
    W2 = params["W2"]  # [n_y,n_h]

    dA2 = -(Y / A2) + (1 - Y) / (1 - A2)  # [n_y,m]
    dZ2 = dA2 * sigmoid_backward(A2)  # [n_y,m]

    dW2 = np.dot(dZ2, A1.T) / m  #  [n_y,n_h] = [n_y,m] * [m,n_h]
    db2 = np.mean(dZ2, axis=1, keepdims=True)  # [n_y,1]

    dA1 = np.dot(W2.T, dZ2)  # [n_h,m] =  [n_h,n_y]* [n_y,m]
    dZ1 = dA1 * sigmoid_backward(A1)  # [n_h,m]

    dW1 = np.dot(dZ1, X.T) / m  # [n_h,n] = [n_h,m] * [ m,n]
    db1 = np.mean(dZ1, axis=1, keepdims=True)  # [n_h,1]

    return dict(dW1=dW1, db1=db1, dW2=dW2, db2=db2)


# Update weights
def update_weights(params: dict, grads: dict, learning_rate: float) -> dict:

    W1, W2 = params["W1"], params["W2"]
    b1, b2 = params["b1"], params["b2"]

    dW1, dW2 = grads["dW1"], grads["dW2"]
    db1, db2 = grads["db1"], grads["db2"]

    # update weights
    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2

    # update bias
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    params["W1"] = W1
    params["W2"] = W2
    params["b1"] = b1
    params["b2"] = b2

    return params


def train(
    X: np.ndarray, y: np.ndarray, n_h: int = 3, epoch: int = 100, learning_rate: float = 0.01
) -> dict:

    n_x = X.shape[0]
    n_y = y.shape[0]
    params = init_parameters(n_x=n_x, n_h=n_h, n_y=n_y)

    print(f"X.shape = {X.shape}, Y.shape = {Y.shape}")
    print(f"n_x = {n_x}, n_h = {n_h}, n_y = {n_y}")
    print(f"initial params: {params}")
    for i in range(epoch):
        A, cache = forward_prop(X=X, params=params)  # forward prop to get prediction
        cost = compute_cost(y=Y, y_hat=A)  # compute cost
        grads = compute_grads(X=X, Y=Y, cache=cache, params=params)  # compute gradient
        params = update_weights(
            params=params, grads=grads, learning_rate=learning_rate
        )  # update weight using gradient descent
        if i % 100 == 0:
            print(f"epoch={i}\tcost={cost}")
    print(f"learnt params: {params}")

    return params


def load_data():

    # X = np.random.randn(2, 100)  # [n,m]. All training examples aligned in columns.
    # Y = np.random.randint(0, 2, size=(1, X.shape[1]))

    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df = pd.read_csv("../data/haberman.data", header=None)
    df[3] = df[3].map({2: 0, 1: 1})  # 1 = survived, 2 = died

    X = scaler.fit_transform(df[[0, 1, 2]].values).T
    Y = df[3].values.reshape(1, -1)

    return X, Y


if __name__ == "__main__":

    X, Y = load_data()
    params = train(X=X, y=Y, n_h=10, epoch=10000, learning_rate=0.001)

    y_pred, _ = forward_prop(X, params)
    y_pred = y_pred.round()

    from sklearn.metrics import accuracy_score

    acc = np.round(accuracy_score(y_true=Y.T, y_pred=y_pred.T), 2) * 100

    print(f"accuracy = {acc}%")


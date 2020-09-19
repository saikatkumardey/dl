import numpy as np

np.random.seed(1)


# Define parameters
def init_parameters(layer_dims: list) -> dict:
    """
    layer_dims: number of neurons in each layer (including the input/output layer)

    returns: a dict of parameters Ws & bs
    """

    params = {}
    for l in range(1, len(layer_dims)):
        params[l] = {
            "W": np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01,
            "b": np.zeros((layer_dims[l], 1)),
        }
    return params


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
    return -np.mean([y * np.nan_to_num(np.log(y_hat)) + (1 - y) * np.nan_to_num(np.log(1 - y_hat))])


# Forward propagation
def forward_prop(X: np.ndarray, params: dict) -> [np.ndarray, dict]:
    cache = {}
    layers = len(params)
    cache[0] = {"A": X}
    for l in range(1, layers + 1):
        Wl = params[l]["W"]
        bl = params[l]["b"]
        Zl = np.dot(Wl, cache[l - 1]["A"]) + bl
        Al = sigmoid(Zl)
        cache[l] = {f"A": Al}
    return Al, cache


# Backward propagation: Compute gradients
def compute_grads(X: np.ndarray, Y: np.ndarray, cache: dict, params: dict) -> dict:

    m = X.shape[0]
    grads = {}
    layers = len(params)

    A = cache[layers]["A"]
    dA = -(Y / A) + (1 - Y) / (1 - A)
    dZ = dA * sigmoid_backward(A)
    grads[layers] = {"dA": dA, "dZ": dZ}

    for l in range(layers - 1, 0, -1):
        next_layer = l + 1
        dA = np.dot(params[next_layer]["W"].T, grads[next_layer]["dZ"])
        dZ = dA * sigmoid_backward(cache[l]["A"])
        grads[l] = {"dA": dA, "dZ": dZ}

    for l in range(1, layers + 1):
        grads[l]["dW"] = np.dot(grads[l]["dZ"], cache[l - 1]["A"].T) / m
        grads[l]["db"] = np.mean(grads[l]["dZ"], axis=1, keepdims=True)

    return grads


# Update weights
def update_weights(params: dict, grads: dict, learning_rate: float) -> dict:

    layers = len(params)
    # update weights & biases
    for l in range(1, layers + 1):
        params[l]["W"] = params[l]["W"] - learning_rate * grads[l]["dW"]
        params[l]["b"] = params[l]["b"] - learning_rate * grads[l]["db"]

    return params


def train(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layers: int = 1,
    n_h: int = 3,
    epoch: int = 100,
    learning_rate: float = 0.01,
) -> dict:

    n_x = X.shape[0]
    n_y = y.shape[0]

    layer_dims = [n_x] + [n_h] * hidden_layers + [n_y]
    params = init_parameters(layer_dims=layer_dims)

    print(f"X.shape = {X.shape}, Y.shape = {Y.shape}")
    print("layer dims", layer_dims)
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
    params = train(X=X, y=Y, hidden_layers=1, n_h=3, epoch=1000, learning_rate=0.0001)

    y_pred, _ = forward_prop(X, params)
    y_pred = y_pred.round()

    from sklearn.metrics import accuracy_score

    acc = np.round(accuracy_score(y_true=Y.T, y_pred=y_pred.T), 2) * 100

    print(f"accuracy = {acc}%")


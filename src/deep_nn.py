import numpy as np
from sklearn.metrics import accuracy_score
from utils import compute_cost, load_data, sigmoid, sigmoid_prime, baseline

np.random.seed(1)


def init_parameters(layer_dims: list) -> dict:
    """
    Initialise parameters (weights/biases) to be learnt by the neural network.

    Parameters
    ----------
    layer_dims: number of neurons in each layer (including the input/output layer)

    Returns
    ---------
    a dict of parameters Ws & bs for each layer
    """

    params = {}
    for l in range(1, len(layer_dims)):
        params[l] = {
            "W": np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01,
            "b": np.zeros((layer_dims[l], 1)),
        }
    return params


def forward_prop(X: np.ndarray, params: dict) -> [np.ndarray, dict]:
    """
    Forward propagation step

    Parameters
    ----------
    X: Input matrix. [n,m] dimension where n is number of features and m is the number of samples.
    params: Dictionary of parameters containing weights and biases for each layer

    Returns
    --------
    a tuple (Prediction of the neural network, Dictionary containing intermediate variables for use in backpropagation)
    """
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


def compute_grads(X: np.ndarray, Y: np.ndarray, cache: dict, params: dict) -> dict:
    """
    Compute gradients using backpropagation algorithm

    Parameters
    ----------
    X: [n,m] matrix of training examples
    Y: [1,m] matrix of output labels/values
    cache: Dictionary of intermediate values from forward-propagation step
    params: Dictionary of weights & biases from each layer

    Returns
    ---------
    a dictionary containing the gradients computed for each layer
    """

    m = X.shape[0]
    grads = {}
    layers = len(params)

    A = cache[layers]["A"]
    dA = -(Y / A) + (1 - Y) / (1 - A)
    dZ = dA * sigmoid_prime(A)
    grads[layers] = {"dA": dA, "dZ": dZ}

    for l in range(layers - 1, 0, -1):
        next_layer = l + 1
        dA = np.dot(params[next_layer]["W"].T, grads[next_layer]["dZ"])
        dZ = dA * sigmoid_prime(cache[l]["A"])
        grads[l] = {"dA": dA, "dZ": dZ}

    for l in range(1, layers + 1):
        grads[l]["dW"] = np.dot(grads[l]["dZ"], cache[l - 1]["A"].T) / m
        grads[l]["db"] = np.mean(grads[l]["dZ"], axis=1, keepdims=True)

    return grads


def update_parameters(params: dict, grads: dict, learning_rate: float) -> dict:
    """
    Update weights & biases using gradient descent.

    Parameters
    ----------
    params: dictionary of weights & biases for each layer
    grads: dictionary of gradients of weights & biases computed from back-propagation
    learning_rate: step-size for gradient descent update

    Returns
    --------
    a dictionary of updated parameters (weights & biases for each layer)
    """

    layers = len(params)
    # update weights & biases
    for l in range(1, layers + 1):
        params[l]["W"] = params[l]["W"] - learning_rate * grads[l]["dW"]
        params[l]["b"] = params[l]["b"] - learning_rate * grads[l]["db"]

    return params


def train(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layer_dims: list = [3],
    epoch: int = 100,
    learning_rate: float = 0.01,
) -> dict:
    """
    Train a neural network

    Parameters
    ----------
    X: [n,m] matrix of training examples
    Y: [1,m] matrix of output labels/values
    hidden_layer_dims: a list of hidden layer dimensions where each item denotes number of neurons in that layer
    epoch: number of iterations to perform
    learning_rate: step-size for gradient descent update

    Returns
    ---------
    a dictionary of learnt parameters (weights & biases)
    """

    n_x = X.shape[0]
    n_y = y.shape[0]

    layer_dims = [n_x] + hidden_layer_dims + [n_y]
    params = init_parameters(layer_dims=layer_dims)

    print(f"X.shape = {X.shape}, Y.shape = {Y.shape}")
    print("layer dims", layer_dims)
    print(f"initial params: {params}")
    for i in range(epoch):
        A, cache = forward_prop(X=X, params=params)  # forward prop to get prediction
        cost = compute_cost(Y=Y, Y_hat=A)  # compute cost
        grads = compute_grads(X=X, Y=Y, cache=cache, params=params)  # compute gradient
        params = update_parameters(
            params=params, grads=grads, learning_rate=learning_rate
        )  # update parameters using gradient descent
        if i % 100 == 0:
            print(f"epoch={i}\tcost={cost}")
    print(f"learnt params: {params}")

    return params


if __name__ == "__main__":

    X, Y = load_data()

    params = train(X=X, y=Y, hidden_layer_dims=[3, 3], epoch=1000, learning_rate=0.0001)
    y_pred, _ = forward_prop(X, params)
    y_pred = y_pred.round()  # prediction > 0.5 is rounded to 1, otherwise 0

    acc = np.round(accuracy_score(y_true=np.ravel(Y), y_pred=np.ravel(y_pred)), 3) * 100
    acc_lib = baseline(X.T, np.ravel(Y))
    print(f"accuracy [our implementation] = {acc}%")
    print(f"accuracy [sklearn] = {acc_lib}%")

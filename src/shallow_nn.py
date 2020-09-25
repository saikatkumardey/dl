import numpy as np
from sklearn.metrics import accuracy_score

from utils import baseline, compute_cost, load_data, sigmoid, sigmoid_prime

np.random.seed(1)


def init_parameters(n_x: int, n_h: int, n_y: int = 1) -> dict:
    """
    Initialise parameters (weights/biases) to be learnt by the neural network.

    Parameters
    ----------
    n_x : size of input layer (number of features)
    n_h : size of hidden layer (number of hidden neurons)
    n_y : size of output layer (number of output neurons)

    Returns
    ----------
    a dict of parameters W & b
    """
    return {
        "W1": np.random.randn(n_h, n_x) * 0.01,  # one weight per feature,
        "W2": np.random.randn(n_y, n_h) * 0.01,
        "b1": np.zeros((n_h, 1)),  # one bias
        "b2": np.zeros((n_y, 1)),
    }


def forward_prop(X: np.ndarray, params: dict) -> [np.ndarray, dict]:
    """
    Forward propagation step

    Parameters
    ----------
    X: Input matrix. [n,m] dimension where n is number of features and m is the number of samples.
    params: Dictionary of parameters containing weights and biases

    Returns
    --------
    a tuple (Prediction of the neural network, Dictionary containing intermediate variables for use in backpropagation)
    """
    W1 = params["W1"]  # [n_h,n_x]
    b1 = params["b1"]  # [n_h,1]

    W2 = params["W2"]  # [n_y,n_h]
    b2 = params["b2"]  # [n_y,1]

    Z1 = np.dot(W1, X) + b1  # [n_h,m] = [n_h,n_x] . [n_x,m] + [1,m]
    A1 = sigmoid(Z1)  # [n_h,m]
    Z2 = np.dot(W2, A1) + b2  # [n_y,m] = [n_y,n_h] . [n_h,m] + [n_y,1]
    A2 = sigmoid(Z2)  # [n_y,m]

    cache = dict(A1=A1, A2=A2)

    return A2, cache


def compute_grads(X: np.ndarray, Y: np.ndarray, cache: dict, params: dict) -> dict:
    """
    Compute gradients using backpropagation algorithm

    Parameters
    ----------
    X: [n,m] matrix of training examples
    Y: [1,m] matrix of output labels/values
    cache: Dictionary of intermediate values from forward-propagation step
    params: Dictionary of weights & biases

    Returns
    ---------
    a dictionary containing the gradients
    """

    m = X.shape[0]
    A1 = cache["A1"]  # [n_h,m]
    A2 = cache["A2"]  # [n_y,m]
    W2 = params["W2"]  # [n_y,n_h]

    dA2 = -(Y / A2) + (1 - Y) / (1 - A2)  # [n_y,m]
    dZ2 = dA2 * sigmoid_prime(A2)  # [n_y,m]

    dW2 = np.dot(dZ2, A1.T) / m  #  [n_y,n_h] = [n_y,m] . [m,n_h]
    db2 = np.mean(dZ2, axis=1, keepdims=True)  # [n_y,1]

    dA1 = np.dot(W2.T, dZ2)  # [n_h,m] =  [n_h,n_y] . [n_y,m]
    dZ1 = dA1 * sigmoid_prime(A1)  # [n_h,m]

    dW1 = np.dot(dZ1, X.T) / m  # [n_h,n] = [n_h,m] . [ m,n]
    db1 = np.mean(dZ1, axis=1, keepdims=True)  # [n_h,1]

    return dict(dW1=dW1, db1=db1, dW2=dW2, db2=db2)


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
    a dictionary of updated parameters (weights & biases)
    """

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
    X: np.ndarray,
    y: np.ndarray,
    n_h: int = 3,
    epoch: int = 100,
    learning_rate: float = 0.01,
) -> dict:
    """
    Train a neural network

    Parameters
    ----------
    X: [n,m] matrix of training examples
    Y: [1,m] matrix of output labels/values
    n_h: size of hidden layer
    epoch: number of iterations to perform
    learning_rate: step-size for gradient descent update

    Returns
    ---------
    a dictionary of learnt parameters (weights & biases)
    """
    n_x = X.shape[0]
    n_y = y.shape[0]
    params = init_parameters(n_x=n_x, n_h=n_h, n_y=n_y)

    print(f"X.shape = {X.shape}, Y.shape = {Y.shape}")
    print(f"n_x = {n_x}, n_h = {n_h}, n_y = {n_y}")
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
    params = train(X=X, y=Y, n_h=10, epoch=10000, learning_rate=0.001)

    y_pred, _ = forward_prop(X, params)
    y_pred = y_pred.round()

    acc = np.round(accuracy_score(y_true=np.ravel(Y), y_pred=np.ravel(y_pred)), 3) * 100
    acc_lib = baseline(X.T, np.ravel(Y))
    print(f"accuracy [our implementation] = {acc}%")
    print(f"accuracy [sklearn] = {acc_lib}%")

# z = w_Tx + b = w1x1 + w2x2 + ..... + wnxn + b
# a = sigmoid(z) #= 0.5 at 0, 1 for very large values, 0 for very small (or negative) values
# L(a,y) = - [yloga + (1-y)log(1-a)], when y=1, L(a,y)= -loga, when y=0, L(a,y) = log(1-a)

# back-prop
# Compute derivate of L with respect to all inputs and intermediate variables
# Apply Gradient descent
# dL/da = - [y/a - (1-y)/(1-a)] # sign changes due to derivative of (1-a) wrt a which is -1
# dL/dz = dL/da * da/dz =- [y/a - (1-y)/(1-a)] * a*(1-a) = - [y(1-a) - (1-y)*a] = - [ y-ay - a + ay] = a-y
# da/dz = a * (1-a) =  # plug it into the previous equation
# dL/dw1 = dL/dz * dz/dw1 =  dL/da * da/dz * dz/dw1, similarly do it for w2,w3...wn
# dz/dw1 = x1, dz/dw2 = x2... # plug it into previous equation
# dz/db = 1
# dL/db = dL/dz * dz/db =  dL/da * da/dz * dz/db =  dL/da * da/dz  (plug stuff in to compute)
# now apply gradient descent, w_new = w_old - alpha * dL/dw, b_new = b_old - alpha * dL/db
import numpy as np

np.random.seed(1)


# Define parameters
def init_parameters(n_x: int) -> dict:
    """
    n_x : number of features

    returns: a dict of parameters W & b
    """
    return {
        "W": np.zeros((1, n_x)),  # one weight per feature,
        "b": np.zeros((1, 1)),  # one bias
    }


# Define activation function
def sigmoid(z: np.array) -> np.array:
    """
    computes sigmoid of z
    """
    return 1 / (1 + np.exp(-z))


# Compute Cost
def compute_cost(y: np.array, y_hat: np.array) -> float:
    return -np.mean([y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)])


# Forward propagation
def forward_prop(X: np.ndarray, params: dict) -> np.ndarray:
    W = params["W"]  # [1,n]
    b = params["b"]  # 0
    Z = np.dot(W, X) + b  # [1,m] = [1,n] * [n,m] + [1,m]
    A = sigmoid(Z)  # [1,m]
    return A


# Backward propagation


# Compute gradients
def compute_grads(X, Y, A, params):

    dA = -(Y / A) + (1 - Y) / (1 - A)  # [1,m]
    dZ = dA * A * (1 - A)  # [1,m]
    dW = np.dot(dZ, X.T)  # [1,n] = [1,m] * [m,n]
    db = np.mean(dZ, axis=1, keepdims=True)  # [1,1]

    return dict(dA=dA, dZ=dZ, dW=dW, db=db)


# Update weights
def update_weights(params: dict, grads: dict, learning_rate: float) -> dict:

    W = params["W"]
    b = params["b"]

    dW = grads["dW"]
    db = grads["db"]

    W = W - learning_rate * dW  # update weights
    b = b - learning_rate * db  # update bias

    params["W"] = W
    params["b"] = b

    return params


def train(X: np.ndarray, y: np.ndarray, epoch: int = 100, learning_rate: float = 0.01) -> dict:

    params = init_parameters(X.shape[0])

    print(f"X.shape = {X.shape}, Y.shape = {Y.shape}")
    print(f"initial params: {params}")
    for i in range(epoch):
        A = forward_prop(X=X, params=params)  # forward prop to get prediction
        cost = compute_cost(y=Y, y_hat=A)  # compute cost
        grads = compute_grads(X=X, Y=Y, A=A, params=params)  # compute gradient
        params = update_weights(
            params=params, grads=grads, learning_rate=learning_rate
        )  # update weight using gradient descent
        if i % 10 == 0:
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


def baseline(X, Y):

    # standard library
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    lr.fit(X, Y)
    y_pred_lib = lr.predict(X)
    acc_lib = accuracy_score(y_true=Y, y_pred=y_pred_lib)
    print(f"accuracy lib = {acc_lib}%")


if __name__ == "__main__":
    X, Y = load_data()
    params = train(X=X, y=Y, epoch=1000, learning_rate=0.01)

    y_pred = forward_prop(X, params).round()

    from sklearn.metrics import accuracy_score

    acc = np.round(accuracy_score(y_true=Y.T, y_pred=y_pred.T), 2) * 100

    print(f"accuracy = {acc}%")

    baseline(X.T, Y.T)


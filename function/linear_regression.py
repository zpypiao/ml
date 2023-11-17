import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from read_data import read_data


# creat the ten-fold data verify model
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# read data
data = read_data()

# calculate the loss partial
def loss_partial(W, b, X, y, alpha):
    
    # calculate the W loss partial
    W_l = np.dot(np.dot(W, X) + b - y, X)

    # calculate the b loss partial
    b_l = np.dot(W, X) + b - y

    # update the value of function
    W -= alpha*W_l
    b -= alpha*b_l

    # return the new value of W and b
    return W, b

# define the process
def train_model(train_data, test_data, alpha):


    # define the initial parameter of function
    W = np.array([0. for _ in range(8)])
    b = 0
    # train the model
    for _ in range(10):
        for each in train_data:
            X = each[:8]
            y = each[8]
            W, b = loss_partial(W, b, X, y, alpha)

    return W, b


# define the outcome of test
def test_model(test_data, W, b):

    Y = []
    Yt = []

    for each in test_data:
        Y.append(each[8])
        Yt.append(np.dot(W, each[:8]) + b)


    plt.plot(Y, 'red')
    plt.plot(Yt, 'blue')
    plt.show()



# define the learn rate
alpha = 0.01

for _, (train_ind, test_ind) in enumerate(kf.split(data), 1):
    train_data = data[train_ind]
    test_data = data[test_ind]

    W, b = train_model(train_data, test_data, alpha)
    test_model(test_data, W, b)
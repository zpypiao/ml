import numpy as np
import matplotlib.pyplot as plt

raw_data = []
# import data
with open('./data/abalone.data', 'r') as file:
    for line in file:
        raw_data.append(line.strip().split(','))

processed_data = []

for each in raw_data:
    if each[0] == 'M':
        each[0] = 1
    elif each[0] == 'F':
        each[0] = 2
    else:
        each[0] = 3
    
    processed_data.append(each)

data = np.asarray(processed_data, dtype=float)


train_data = data[len(data)//10:]
test_data = data[:len(data)//10]


# define the initial parameter of function
W = np.array([0. for _ in range(8)])
b = 0

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

# define the learn rate
alpha = 0.001

# train the model
for each in train_data:
    X = each[:8]
    y = each[8]
    W, b = loss_partial(W, b, X, y, alpha)


# # test the data
# X = test_data[:][:8]
# Y = test_data[:][8]
# Yt = np.dot(W, X) + b

Y = []
Yt = []

for each in test_data:
    Y.append(each[8])
    Yt.append(np.dot(W, each[:8]) + b)

print(Y, Yt)

plt.plot(Y, 'red')
plt.plot(Yt, 'blue')
plt.show()
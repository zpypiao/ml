import numpy as np


x = np.array([[1, 2, 3], [1, 4, 5]])


# 创建一个数组
arr = np.array([1, 2, 3])

# 计算指数值
exp_arr = np.exp(arr)

print("原始数组:")
print(arr)

print("\n指数值数组:")
print(exp_arr)


print(np.max(x, axis=1))


mydc = {}

mydc['asa'] = 1

mydc[2] = 3

print(mydc[2])


def plot_tree(node, label, current_index=97, current_str=''):
    string = ''
    left = False
    right = False
    next_ind = current_index + 1
    if node.left.value:
        string += chr(current_index) + '[' + label[node.feature] + ']-->|<' + \
            str(node.threshold) + '|' + chr(next_ind) + '[' + node.left.result + ']//'
        next_ind += 1
        left = True
    if node.right.value:
        string += chr(current_index) + '[' + label[node.feature] + ']-->|>' + \
            str(node.threshold) + '|' + chr(next_ind) + '[' + node.right.result + ']//'
        next_ind += 1
        right = True
    if not left:
        string += chr(current_index) + '[' + label[node.feature] + ']-->|<' + \
            str(node.threshold) + '|' + chr(next_ind) + '[' + label[node.left.feature] + ']//'
        add_string, next_ind= plot_tree(node.left, label, next_ind)
        string += add_string
    if not right:
        string += chr(current_index) + '[' + label[node.feature] + ']-->|>' + \
            str(node.threshold) + '|' + chr(next_ind) + '[' + label[node.right.feature] + ']//'
        add_string, next_ind= plot_tree(node.left, label, next_ind)
        string += add_string
    return string, next_ind



print(type(chr(65)))


import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # 减去最大值以提高数值稳定性
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def initialize_parameters(self, num_features, num_classes):
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros((1, num_classes))

    def train(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        self.initialize_parameters(num_features, num_classes)

        for epoch in range(self.num_epochs):
            # Forward pass
            logits = np.dot(X, self.weights) + self.bias
            probabilities = self.softmax(logits)

            # Compute cross-entropy loss
            one_hot_labels = np.eye(num_classes)[y]
            loss = -np.sum(one_hot_labels * np.log(probabilities)) / num_samples

            # Backward pass (gradient descent)
            gradient_weights = np.dot(X.T, probabilities - one_hot_labels) / num_samples
            gradient_bias = np.sum(probabilities - one_hot_labels, axis=0, keepdims=True) / num_samples

            # Update parameters
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

            # Print loss for every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        probabilities = self.softmax(logits)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

# 示例数据
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 1, 0, 1])

# 创建并训练 softmax 回归模型
model = SoftmaxRegression(learning_rate=0.01, num_epochs=1000)
model.train(X_train, y_train)

# 示例预测
X_test = np.array([[1, 2], [4, 5]])
predictions = model.predict(X_test)
print("Predictions:", predictions)


import numpy as np

class SoftmaxRegressionWithRegularization:
    def __init__(self, learning_rate=0.01, regularization_strength=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def initialize_parameters(self, num_features, num_classes):
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros((1, num_classes))

    def train(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        self.initialize_parameters(num_features, num_classes)

        for epoch in range(self.num_epochs):
            # Forward pass
            logits = np.dot(X, self.weights) + self.bias
            probabilities = self.softmax(logits)

            # Compute cross-entropy loss with L2 regularization
            one_hot_labels = np.eye(num_classes)[y]
            loss = -np.sum(one_hot_labels * np.log(probabilities + 1e-10)) / num_samples
            regularization_term = 0.5 * self.regularization_strength * np.sum(self.weights**2)
            loss += regularization_term

            # Backward pass (gradient descent with L2 regularization)
            gradient_weights = (np.dot(X.T, probabilities - one_hot_labels) + self.regularization_strength * self.weights) / num_samples
            gradient_bias = np.sum(probabilities - one_hot_labels, axis=0, keepdims=True) / num_samples

            # Update parameters
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

            # Print loss for every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        probabilities = self.softmax(logits)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

# 示例数据
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 1, 0, 1])

# 创建并训练带有 L2 正则化的 softmax 回归模型
model_with_regularization = SoftmaxRegressionWithRegularization(learning_rate=0.01, regularization_strength=0.01, num_epochs=1000)
model_with_regularization.train(X_train, y_train)

# 示例预测
X_test = np.array([[1, 2], [4, 5]])
predictions_with_regularization = model_with_regularization.predict(X_test)
print("Predictions with Regularization:", predictions_with_regularization)

import os, requests
import numpy as np
import pandas as pd
from pathlib import Path
from softmax import load_data
from collections import Counter
from sklearn.model_selection import train_test_split


class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=False, result=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.result = result

class Tree:

    # 初始化决策树
    def __init__(self, method=None):
        self.root = None
        
        if method == 'C4.5':
            self.method = 'C4.5'
        elif method == 'CART':
            self.method = 'CART'
        else:
            print('Error: the method is not support')
        pass

    # 递归构建决策树
    def build_tree(self, X, y, depth=0, max_depth=5):

        # 如果达到最大深度，则返回以概率最高为结果的节点
        if depth == max_depth:
            result = np.bincount(y).argmax()
            return Node(value=True, result=result)
        
        # 如果仅有一个值，则不必继续切分
        if len(y) == 1:
            return Node(value=True, result=y[0])
        
        best_feature, best_threshold = self.find_best_class(X, y)

        if best_feature:
            # 获取锚点左侧索引
            index_left = X[:, best_feature] <= best_threshold

            # 创建左右节点
            left = self.build_tree(X[index_left], y[index_left], depth=depth+1, max_depth=max_depth)

            # 创建右节点
            right = self.build_tree(X[~index_left], y[~index_left], depth=depth+1, max_depth=max_depth)

            # 返回创建的节点
            node = Node(feature=best_feature, threshold=best_threshold, left=left, right=right)
        else:
            result = np.bincount(y).argmax()
            return Node(value=True, result=result)

        # 记录根节点
        if depth == 0:
            self.root = node

        return node

    # 找到最优切分点
    def find_best_class(self, X, y):
        
        # 初始化
        best_feature = None
        best_threshold = None
        best_outcome = float('-inf')
        features = X.shape[1]

        # 遍历所有特征
        for feature in range(features):

            # 找到所有切分点
            single_value = np.unique(X[:, feature])
            thresholds = (single_value[:-1] + single_value[1:]) / 2

            # 遍历所有threshold
            for th in thresholds:
                
                additional = self.additional_value(X, y, feature, th)

                if additional > best_outcome:

                    best_feature = feature
                    best_threshold = th
                    best_outcome = additional

        return best_feature, best_threshold

    # 定义信息增益
    def additional_value(self, X, y, feature, th):

        # 根据初始化选择的模式选择不同的信息计算函数
        if self.method == 'C4.5':
            func = self.entropy
        else:
            func = self.gini

        # 获取锚点左侧索引
        index_left = X[:, feature] <= th

        # 划分y值
        y_left, y_right = y[index_left], y[~index_left]

        parent_value = func(y)

        left_value = func(y_left)

        right_value = func(y_right)

        # 计算信息增益
        additional = parent_value - 1.0*(len(y_left)/len(y))*left_value - 1.0*(len(y_right)/len(y))*right_value

        return additional


    # 计算信息熵
    def entropy(self, label):
        num = len(label)
        unilabel = np.unique(label)
        unidict = dict(Counter(label))

        result = 0

        for each in unilabel:
            prob = 1.0*unidict[each] / num
            result -= prob*np.log2(prob)

        return result
    
    # 计算基尼指数
    def gini(self, label):
        num = len(label)
        unilabel = np.unique(label)
        unidict = dict(Counter(label))
        result = 0
        for each in unilabel:
            result += (unidict[each] / num) ** 2

        return 1 - result
        # return result


    # 定义决策树的预测
    def predict(self, X_test):
        y_predict = []
        for each in X_test:
            y_predict.append(self.single_predict(self.root, each))

        return np.array(y_predict)
    

    def single_predict(self, node, x):

        # 如果value为True，则为叶子节点，直接输出结果
        if node.value:
            return node.result
        
        # 左右节点分流
        elif x[node.feature] <= node.threshold:
            return self.single_predict(node.left, x)
        
        else:
            return self.single_predict(node.right, x)

# 打印决策树的结构
# 该打印结果适用于mermaid插件
# 将打印结果粘贴在mermaid编辑框中，即可显示其结构
def plot_tree(node, label, current_index=0, current_str=[]):
    left = False
    right = False
    next_ind = current_index + 1
    if node.left.value:
        current_str.append('A' + str(current_index)  + '-->|<' + \
            str(node.threshold) + '|' + 'A' + str(next_ind) + '[' + str(node.left.result) + ']')
        next_ind += 1
        left = True
    if node.right.value:
        current_str.append('A' + str(current_index) + '-->|>' + \
            str(node.threshold) + '|' + 'A' + str(next_ind) + '[' + str(node.right.result) + ']')
        next_ind += 1
        right = True
    if not left:
        current_str.append('A' + str(current_index) +  '-->|<' + \
            str(node.threshold) + '|' + 'A' + str(next_ind) + '[' + str(label[node.left.feature]) + ']')
        _, next_ind= plot_tree(node.left, label, next_ind, current_str)
    if not right:
        current_str.append('A' + str(current_index) + '-->|>' + \
            str(node.threshold) + '|' + 'A' + str(next_ind) + '[' + str(label[node.right.feature]) + ']')
        _, next_ind= plot_tree(node.right, label, next_ind, current_str)

    return current_str, next_ind

if __name__ == '__main__':

    data = load_data(url='http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    train_X = np.array(train_data.drop('quality', axis=1))
    train_label = np.array(train_data['quality'])

    test_X = np.array(test_data.drop('quality', axis=1))
    test_label = np.array(test_data['quality'])

    decision_tree = Tree('CART')

    decision_tree.build_tree(train_X, train_label, max_depth=4)

    yp = decision_tree.predict(X_test=test_X)

    corr = np.sum(yp == test_label)

    print(corr/len(yp))

    # 打印出决策树的结构
    for each in plot_tree(decision_tree.root, data.columns)[0]:
        print(each)
    print(data.columns[decision_tree.root.feature])
import os, requests
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# 下载文件
def load_data(url=None, file_path=None):

    # 有文件则读文件
    if file_path:

        # 检查文件目录
        if not os.path.exists(file_path):
            return None
        
        # 返回文件数据
        return pd.read_csv(file_path, sep=';')
    
    # 如果是url，则下载文件到data目录
    file_path = Path(__file__).parent / '../data/winequality-white.csv'

    file = requests.get(url)

    with open(file_path, 'wb') as f:
        f.write(file.content)

    return pd.read_csv(file_path, sep=';')

# one hot编码
def one_hot(label):

    # 获得label数量
    label_num = label.size

    # 获得唯一label数量
    unique_label = len(np.unique(label))

    # 创建映射字典
    one_hot_dict = {}
    i = 0
    for each in np.unique(label):
        one_hot_dict[each] = i
        i += 1

    # 创建数组
    result = np.zeros((label_num, unique_label))

    # 进行编码
    for i in range(label_num):
        result[i, one_hot_dict[label[i]]] = 1

    return result

# 创建softmax方法
def softmax(logits):

    # 减去最大值以提高数值稳定性
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    
    # 计算结果
    result = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return result



# 创建softmax方法
def softmax_model(X, label, iter=1000, alpha=0.1, lam=0.01):

    # 获得输入数据数量和特征数量
    sample_num, feature_num = X.shape

    # 获得分类数量
    class_num = label.shape[1]

    # 初始化权重矩阵
    w = np.zeros((feature_num, class_num))

    # 记录损失函数
    Loss = []
    
    # 开始计算
    for _ in range(iter):

        # 计算结果
        sc = np.dot(X, w)

        # 计算softmax值
        # 因为存在0，所以计算损失和梯度时会报inf， 因此增加偏移量
        pro = softmax(sc) + 1e-10

        # 计算损失函数
        loss = -(1.0/sample_num)*np.sum(label*np.log(pro))
 

        Loss.append(loss)

        # 计算梯度
        gradient = -(1.0/sample_num)*(np.dot(X.T, (label-pro)) + lam*w)
        
        # 更新权重
        w -= alpha*gradient

    return w, Loss

def predict(X, w):

    sc = np.dot(X, w)
    pro = softmax(sc)

    return np.argmax(pro, axis=1).reshape((-1,1))


if __name__ == '__main__':
    data = load_data(url='http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')

    X = np.array(data.drop('quality', axis=1))
    label = one_hot(data['quality'])

    weight, loss = softmax_model(X, label)

    yp = predict(X, weight)
    corr = np.sum(yp == label)
    acc = corr/len(yp)

    print(acc)
    print(np.sum(weight, axis=1))

    for i in weight:
        print('| ', end = '')
        for j in i:
            print(j, end=' |')
        print('\n')
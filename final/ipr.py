from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# 设置中文字体
matplotlib.rc('font', family='Microsoft YaHei')

# 读取数据函数及数据处理函数
def read_data(file='./data/abalone.data', mode='None', header_insert=0):
    # 读取数据
    df = pd.read_table(file, header=None, sep=',')

    # 创建数据表头
    df.columns = ['gender', 'length', 'diameter', 'height', 'weight', 'non-shell weight', 'organ weight', 'shell weight', 'rings']

    print(df)
    # 定义性别对应的映射字典
    gender_map = {'F':-1, 'I':0, 'M':1}

    # 替换数据中的性别
    df['gender'] = df['gender'].map(gender_map)

    # 截取非环部分
    data = df.iloc[:, :8]

    if mode == 'standard':
        # 截取非环数部分进行标准化处理
        data = (data - data.mean())/data.std()
    elif mode == 'one':
        # 截取非环数部分进行归一化处理
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data = (data - data_min) / (data_max - data_min)
    elif mode == 'None':
        pass

    # 将年龄添加到最后一列(年龄=环数+1.5)
    data['age'] = df['rings'] + 1.5

    if header_insert:
        # 插入一列全为1的值代表与theta_0相乘的x_0
        data.insert(0, 'x_0', 1)

    # 返回pd形式数据
    return data

# 定义一个基于梯度下降的线性回归函数
def linear(X, y, alpha=0.01, iter = 1000):

    # 初始化参数
    W = np.zeros(X.shape[1])

    for _ in range(iter):
        # 计算损失函数值
        loss = np.dot(X, W) - y

        # 计算梯度
        gradient = 2/len(y) * np.dot(X.T, loss)

        # 更新权重
        W -= alpha*gradient

        # 定义终止条件
        if np.sum(error(W, X, y)) < 1e-5:
            break

    return W

# 计算MSE
def error(W, X, y):
    loss = X.dot(W) - y
    return loss


# 读取数据
url = 'http://archive.ics.uci.edu/ml/datasets/Abalone'
data = read_data(file='abalone.data', mode='standard', header_insert=True)

# 划分数据集、测试集与验证集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
test_data, verify_data = train_test_split(test_data, test_size=0.2, random_state=42)

# 创建自变量的表头
x_ind = ['x_0', 'gender', 'length', 'diameter', 'height', 'weight', 'non-shell weight', 'organ weight', 'shell weight']

# 分离自变量与因变量
train_X = np.array(train_data[x_ind])
train_y = np.array(train_data['age'])
test_X = np.array(test_data[x_ind])
test_y = np.array(test_data['age'])
verify_X = np.array(verify_data[x_ind])
verify_y = np.array(verify_data['age'])



# 获取线性回归的权重结果
W_1 = linear(train_X, train_y, alpha=0.01, iter=1000)

# 获得测试集误差
Error = error(W_1, test_X, test_y)

# 获得输入与误差的线性回归结果
W_2 = linear(test_X, Error)

# 获得测试集的结果
yy = verify_y
yt = verify_X.dot(W_1)*35
yipr = verify_X.dot(W_2)
err = error(W_1, verify_X, verify_y)


# 展示结果
plt.plot(yy, 'r', label='实际值')
plt.plot(yt, 'blue', label = '预测值')
plt.plot(yipr, 'y', label = '预测值的可靠性')
# plt.plot(err, 'k')
plt.legend()
plt.show()
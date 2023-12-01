import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

# 创建计算函数运行时间的装饰函数
def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        cost_time = end_time - start_time
        print(f"{func.__name__} 执行时间：{cost_time:.4f} 秒")
        return result
    return wrapper

# 读取数据函数及数据处理函数
def read_data(file='./data/abalone.data', mode='None', header_insert=0):
    # 读取数据
    df = pd.read_table(file, header=None, sep=',')

    # 创建数据表头
    df.columns = ['gender', 'length', 'diameter', 'height', 'weight', 'non-shell weight', 'organ weight', 'shell weight', 'rings']

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


# 创建lasso函数
def lasso(X, y, lam=0.2, alpha=0.01, iter=1000):
    dimension = X.shape[1]
    W = np.zeros(dimension)

    for _ in range(iter):

        loss = np.dot(X, W) - y

        gradient_W = 2/len(y) * np.dot(X.T, loss) + lam*np.sign(W)


        W -= alpha*gradient_W


    return W

# 创建岭回归函数
def ridge(X, y, lam=0.2, alpha=0.01, iter=1000):

    # 获取自变量维度
    dimension = X.shape[1]

    # 初始化权重
    W = np.zeros(dimension)

    for _ in range(iter):

        # 计算损失
        loss = np.dot(X, W) - y

        # 计算梯度
        gradient_W = 2/len(y) * np.dot(X.T, loss) + 2*lam*W

        # 更新权重
        W -= alpha*gradient_W

    # 返回权重
    return W

# 创建局部加权线性回归函数
def kregression(X, y, test_point, k=1.0):
    # 根据查询点计算权重
    weights = np.exp(-np.sum((X - test_point)**2, axis=1) / (2 * k**2))

    # 计算带权重的 X 和 y
    X_weights = X * weights[:, np.newaxis]
    y_weights = y * weights

    if np.linalg.det(X_weights.T @ X)==0:
        # 奇异矩阵，不能继续操作
        return 0

    # 计算参数 theta
    W = np.linalg.inv(X_weights.T @ X) @ X_weights.T @ y_weights

    # 计算预测值
    yt = test_point @ W.flatten()

    return yt




# 计算MSE
def error(W, X, y):
    loss = X.dot(W) - y
    return 1/len(y)*loss.T.dot(loss)


# 定义10折验证模型，将几种方法集成进来
@timing
def train_model(X, y, func='linear', alpha=0.01, iter=1000, lam=0.2, k=1, export_w=False):
    # 创建局部加权线性回归函数MSE计算
    def k_err(train_x, train_y, test_x, test_y, k):

        # 获得测试集样本数
        num = len(test_y)

        total_err = 0

        for i in range(num):
            
            # 计算误差
            error = kregression(train_x, train_y, test_x[i], k) - test_y[i]

            total_err += error**2

        return total_err/num
    
    # 创建列表存储计算所得的误差
    Error = []

    # 存储最小MSE与对应的权重值
    min_err = 10000
    theta = []

    # 创建10折验证模型
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for _, (train_ind, test_ind) in enumerate(kf.split(X), 1):
        train_X, train_y = X[train_ind], y[train_ind]
        test_X, test_y = X[test_ind], y[test_ind]
        # mse =  k_err(train_X, train_y, test_X, test_y, k=0.5)
        if func == 'linear':
            W = linear(train_X, train_y, alpha=alpha)
        elif func == 'lasso':
            W = lasso(train_X, train_y, alpha=alpha, lam=lam)
        elif func == 'ridge':
            W = ridge(train_X, train_y, alpha=alpha, lam=lam)
        elif func == 'k':
            err = k_err(train_X, train_y, test_X, test_y, k=k)
            Error.append(err)
            continue
        
        # 计算均方误差
        err = np.sum(error(W, test_X, test_y))

        # 判断是否更新最优组合
        if err < min_err:
            min_err = err
            theta = W

        Error.append(err)

    if export_w:
        return Error, theta
    
    return Error

# 定义一个展示函数
def display(W, X, y):
    yt = X.dot(W)
    plt.plot(y, 'red')
    plt.plot(yt, 'blue')
    plt.show()

if __name__ == '__main__':
    al = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    data = read_data(mode='one', header_insert=True)

    x_ind = ['x_0', 'gender', 'length', 'diameter', 'height', 'weight', 'non-shell weight', 'organ weight', 'shell weight']

    # 分离出x与y
    X = np.array(data[x_ind])
    y = np.array(data['age'])


    al = [0.45]
    outcome = []
    # for each in al:
    #     Error = train_model(X, y, data, 'linear', alpha=each)
    #     des = {'min':min(Error), 'avg':np.average(Error)}
    #     outcome.append([each, des])

    # train_x, train_y = X[:3000], y[:3000]
    # test_x, test_y = X[3000:], y[3000:]

    # yt = kregression(train_x, train_y, test_point=test_x[10], k=1.0)



    error = train_model(X, y, func='k', k=1.0)

    print(error)
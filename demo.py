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
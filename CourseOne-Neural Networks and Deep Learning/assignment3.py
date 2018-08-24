import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from DeepLearning.CourseOne.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


# 定义神经网络的结构
def layer_sizes(X, Y):
    n_x = X.shape[0]  # size of input layer
    n_y = Y.shape[0]  # size of output layer
    return n_x, n_y


# 初始化参数W, b
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01  # W为标准正态分布里的随机值
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


# 正向传播
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# compute_cost
def compute_cost(A2, Y):
    m = Y.shape[1]
    cost = -1/m * (np.dot(Y, np.log(A2).T) + np.dot((1-Y), np.log(1-A2).T))
    cost = np.squeeze(cost)
    return cost


# 反向传播
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    # W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


#  更新参数
def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    n_x, n_y = layer_sizes(X, Y)
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return parameters


# 分类预测
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    return predictions



# 本实验用的数据
# X是一个维度为(2,400)的矩阵，矩阵的每一列为一个样本，看做平面上的一个点。
# Y是一个(1,400)的标签向量，0表示红色，1表示蓝色
X, Y = load_planar_dataset()
# 显示数据
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
# plt.show()


'''
# 额外的数据
noisy_circles, noisy_moons, blobs, gaussian_quantiles = load_extra_datasets()
datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}
dataset = "noisy_moons"
X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])
# make blobs binary
if dataset == "blobs":
    Y = Y % 2
# 显示数据
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
# plt.show()
'''


'''
# 用sklearn自带的函数做逻辑回归
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)
# 画出决策边界
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()
# 打印准确率
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' %
      float((np.dot(Y, LR_predictions) + np.dot(1-Y, 1-LR_predictions))/float(Y.size)*100) +
      '% ' + "(percentage of correctly labelled datapoints)")
'''


#  用浅层神经网络进行分类
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)


# 打印准确率
predictions = predict(parameters, X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T)
                              + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + '%')


# 画出决策边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

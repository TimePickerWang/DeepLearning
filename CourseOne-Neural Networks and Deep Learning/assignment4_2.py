import pylab
import matplotlib.pyplot as plt
from DeepLearning.CourseOne.assignment4_1 import *


plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# 显示图片
def show_pic(index, x, y, classes):
    plt.imshow(x[index])
    pylab.show()
    print("It's a '" + classes[np.squeeze(y[:, index])].decode("utf-8") + "' picture.")


#  绘制代价函数曲线
def draw_cost_line(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per 100 iterations)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


# 2层神经网络
def two_layer_model(X, Y, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    n_x = X.shape[0]
    n_h = 7
    n_y = 1
    grads = {}
    costs = []
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(num_iterations):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        # Forward propagation
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        # Compute cost
        cost = compute_cost(A2, Y)
        # Backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        parameters = update_parameters(parameters, grads, learning_rate)  # Update parameters
        # Print the cost every 100 iterations and add the cost to the costs List
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)
    draw_cost_line(np.squeeze(costs), learning_rate)  # 绘制代价函数曲线
    return parameters


# L层神经网络
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)  # Forward propagation
        cost = compute_cost(AL, Y)  # Compute cost
        grads = L_model_backward(AL, Y, caches)  # Backward propagation
        parameters = update_parameters(parameters, grads, learning_rate)  # Update parameters
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)
    # draw_cost_line(np.squeeze(costs), learning_rate)  # 绘制代价函数曲线
    return parameters



# 读数据，训练集有209张图片，测试集有50张图片，图片的RGB信息存储在train_set_x_orig和test_set_x_orig中，
# 每张图片通过RGB这3个通道又分为3个64*64的矩阵存储，即每张图片的维度为（64, 64, 3）。
# train_set_y和test_set_y是图片的类别标签，值为0或1，值为0时表示图片不是猫，为1时表示是猫
# classes是一个内容为[b'non-cat' b'cat']的数组，用来映射train_set_y和test_set_y中的标签。
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


# 显示某张图片
# index = 7
# show_pic(index, train_x_orig, train_y, classes)

# 将每张图片的信息压缩为1列，即原来图片的维度为(64, 64, 3)，现在转换成(64*64*3，1)。
# 然后把所有图片的信息存在一个矩阵里，矩阵的列数为图片的张数。这样原来测试集的
# 数据(209,64,64,3)转换成(64*64*3，209)，即将4位的矩阵转换为了2维。
train_set_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_set_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# 图片RGB的取值范围为0~255，用所有数据除以255来进行标准化
train_x = train_set_x_flatten/255.0
test_x = test_set_x_flatten/255.0




# 用2层神经网络优化参数并进行测试
# parameters = two_layer_model(train_x, train_y, learning_rate=0.0075, num_iterations=2500, print_cost=True)
# predictions_train = predict(train_x, train_y, parameters)
# predictions_test = predict(test_x, test_y, parameters)


# 用L层神经网络优化参数并进行测试
layers_dims = [12288, 20, 7, 5, 1]  # 定义一个5层的神经网络模型
parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.0075, num_iterations=2500, print_cost=True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)

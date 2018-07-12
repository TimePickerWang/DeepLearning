from DeepLearning.CourseTwo.init_utils import *

# Initialization

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# 绘制代价函数曲线
def draw_cost_line(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


# 3层神经网络
def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he", show_cost_line=False):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    show_cost_line -- if True,show the cost line

    Returns:
    parameters -- parameters learnt by the model
    """
    costs = []  # cost value list
    layers_dims = [X.shape[0], 10, 5, 1]

    # 初始化参数的3种方式
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    for i in range(num_iterations):
        a3, cache = forward_propagation(X, parameters)
        cost = compute_loss(a3, Y)
        grads = backward_propagation(X, Y, cache)
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
    if show_cost_line:
        draw_cost_line(costs, learning_rate)  # 绘制代价函数曲线
    return parameters


# Zeros initialization
def initialize_parameters_zeros(layers_dims):
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


# Random initialization
def initialize_parameters_random(layers_dims):
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


# He initialization
def initialize_parameters_he(layers_dims):
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2/layers_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


# 获取数据,参数为True则显示数据
train_X, train_Y, test_X, test_Y = load_dataset(False)

parameters = model(train_X, train_Y, initialization="he", show_cost_line=False)

# print("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters)  # 训练集预测
# print("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)  # 测试集预测

# 绘制决策边界
plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

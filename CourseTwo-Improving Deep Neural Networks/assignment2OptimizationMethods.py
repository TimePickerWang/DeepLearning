import math
from DeepLearning.CourseTwo.opt_utils import *
# from DeepLearning.CourseTwo.testCase2 import *

# Optimization Methods

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# 绘制代价函数
def show_cost_line(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()


# Batch Gradient Descent
def update_parameters_with_gd(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural networks
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
    return parameters


# random mini batches
def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[1]  # number of training examples
    mini_batches = []
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition
    num_complete_minibatches = math.floor(m / mini_batch_size)  # number of full mini batches
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:  # Handling the leftover case (last mini-batch < mini_batch_size)
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# initialize velocity(初始化Momentum算法所需变量)
def initialize_velocity(parameters):
    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    return v


# update parameters with momentum(用Momentum算法更新参数)
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural networks
    for l in range(L):
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        parameters["W" + str(l + 1)] -= learning_rate * v["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * v["db" + str(l + 1)]
    return parameters, v


# initialize adam(初始化Adam算法所需变量)
def initialize_adam(parameters) :
    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
    return v, s


# update parameters with adam(用Adam算法更新参数)
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9,
                                beta2=0.999,  epsilon=1e-8):
    L = len(parameters) // 2  # number of layers in the neural networks
    v_corrected = {}
    s_corrected = {}
    for l in range(L):
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1 ** t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1 ** t)

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * (grads["dW" + str(l + 1)] ** 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * (grads["db" + str(l + 1)] ** 2)
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2 ** t)

        parameters["W" + str(l + 1)] -= \
            learning_rate * (v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] -= \
            learning_rate * (v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))

    return parameters, v, s


# 用3种不同的优化算法更新参数
def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999,  epsilon=1e-8, num_iterations=10000, print_cost=True, shwo_cost=False):
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update

    parameters = initialize_parameters(layers_dims)  # Initialize parameters

    # 初始化优化算法所需参数
    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    for i in range(num_iterations):
        mini_batches = random_mini_batches(X, Y, mini_batch_size)

        for mini_batche in mini_batches:
            mini_batch_X, mini_batch_Y = mini_batche
            A3, cache = forward_propagation(mini_batch_X, parameters)
            cost = compute_cost(A3, mini_batch_Y)
            grads = backward_propagation(mini_batch_X, mini_batch_Y, cache)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate,
                                                         beta1, beta2,  epsilon)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    if shwo_cost:  # 绘制代价函数
        show_cost_line(costs, learning_rate)

    return parameters


# 获取数据
train_X, train_Y = load_dataset()

layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer="adam", shwo_cost=False)


# Predict
# predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

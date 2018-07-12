import pylab
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from DeepLearning.CourseTwo.tf_utils import *

np.random.seed(1)
# Tensorflow Tutorial

def linear_function():
    np.random.seed(1)

    X = tf.constant(np.random.randn(3, 1), name="X")
    W = tf.constant(np.random.randn(4, 3), name="W")
    b = tf.constant(np.random.randn(4, 1), name="b")
    Y = tf.add(tf.matmul(W, X), b)

    with tf.Session() as session:
        result = session.run(Y)
    return result


def sigmoid(z):
    x = tf.placeholder(tf.float32, name="x")
    a = tf.sigmoid(x)
    with tf.Session() as session:
        result = session.run(a, feed_dict={x: z})
    return result


def cost(logits, labels):
    z = tf.placeholder(tf.float32, name="z")
    y = tf.placeholder(tf.float32, name="y")
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    with tf.Session() as session:
        cost = session.run(cost, feed_dict={z: logits, y: labels})
    return cost


def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and
    the jth column corresponds to the jth training example. So if example j had
    a label i. Then entry (i,j) will be 1.

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """
    C = tf.constant(C, name="C")
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    with tf.Session() as session:
        one_hot = session.run(one_hot_matrix)
    return one_hot


def ones(shape):
    ones = tf.ones(shape)
    with tf.Session() as session:
        ones = session.run(ones)
    return ones


###############################################################
def create_placeholders(n_x, n_y):
    X = tf.placeholder(dtype=tf.float32, shape=[n_x, None])
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y, None])
    return X, Y


# 初始化参数
def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    return parameters


# 向前传播
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3


# 计算代价函数
def compute_cost(Z3, Y):
    # "logits" and "labels" inputs of tf.nn.softmax_cross_entropy_with_logits are expected
    #  to be of shape (number of examples, num_classes)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    seed = 3  # to keep consistent results
    tf.set_random_seed(1)  # to keep consistent results

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Initialize all the variables
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        for i in range(num_epochs):
            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = session.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            if print_cost is True and i % 100 == 0:
                print("Cost after epoch %i: %f" % (i, epoch_cost))
            if print_cost is True and i % 5 == 0:
                costs.append(epoch_cost)

        # 绘制代价函数
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = session.run(parameters)
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        return parameters




# 获取数据，训练集有1080张图片，测试集有120张图片，图片的RGB信息存储在X_train_orig和X_test_orig中，
# 每张图片通过RGB这3个通道又分为3个64*64的矩阵存储，即每张图片的维度为（64, 64, 3）。
# Y_train_orig和Y_test_orig是图片的类别标签，值为0、1、2、3、4、5，
# classes是一个内容为[0 1 2 3 4 5]的数组，用来映射Y_train_orig和Y_test_orig中的标签。
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# 显示某张图片
# index = 0
# plt.imshow(X_train_orig[index])
# pylab.show()
# print("y = " + str(np.squeeze(Y_train_orig[:, index])))

# 将每张图片的信息压缩为1列
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# 标准化
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# 将测试集和训练集转为one hot矩阵
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

parameters = model(X_train, Y_train, X_test, Y_test)

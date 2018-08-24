import pylab
import scipy
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from DeepLearning.CourseOne.lr_utils import load_dataset


# 显示图片
def show_pic(index, x, y, classes):
    plt.imshow(x[index])
    pylab.show()
    print("It's a '" + classes[np.squeeze(y[:, index])].decode("utf-8") + "' picture.")


# sigmoid函数
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# 初始化参数
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


#  正向传播和反向传播
def propagate(w, b, X, Y):
    m = X.shape[1]
    # FORWARD PROPAGATION  正向传播
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # compute cost
    # BACKWARD PROPAGATION 反向传播
    dz = A - Y
    dw = 1 / m * np.dot(X, dz.T)
    db = 1 / m * np.sum(dz)

    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw, "db": db}
    return grads, cost


# 优化参数，num_iterations为迭代次数，learning_rate为学习率，print_cost为True时
# 每迭代100次打印一次代价函数值
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    params = {"w": w, "b": b}
    return params, costs


# 分类预测函数
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    assert (Y_prediction.shape == (1, m))
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])
    # Gradient descent
    parameters, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d


#  绘制学习曲线
def draw_cost(d):
    # Plot learning curve (with costs)
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()


# 测试自己的图片
def test_myPic(picname, classes, d):
    dir = "./testImgs/"
    filename = dir + picname
    image = np.array(ndimage.imread(filename, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)
    # 显示图片
    plt.imshow(image)
    pylab.show()
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
        int(np.squeeze(my_predicted_image))].decode("utf-8") + "\" picture.")



# 读数据，训练集有209张图片，测试集有50张图片，图片的RGB信息存储在train_set_x_orig和test_set_x_orig中，
# 每张图片通过RGB这3个通道又分为3个64*64的矩阵存储，即每张图片的维度为（64, 64, 3）。
# train_set_y和test_set_y是图片的类别标签，值为0或1，值为0时表示图片不是猫，为1时表示是猫
# classes是一个内容为[b'non-cat' b'cat']的数组，用来映射train_set_y和test_set_y中的标签。
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 显示某张图片
# index = 1
# show_pic(index, test_set_x_orig, test_set_y, classes)

# 将每张图片的信息压缩为1列，即原来图片的维度为(64, 64, 3)，现在转换成(64*64*3，1)。
# 然后把所有图片的信息存在一个矩阵里，矩阵的列数为图片的张数。这样原来测试集的
# 数据(209,64,64,3)转换成(64*64*3，209)，即将4位的矩阵转换为了2维。
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# 图片RGB的取值范围为0~255，用所有数据除以255来进行标准化
train_set_x = train_set_x_flatten/255.0
test_set_x = test_set_x_flatten/255.0

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

#  绘制学习曲线
# draw_cost(d)

# 测试自己的图片
test_myPic("my_image2.jpg", classes, d)

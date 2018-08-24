import h5py
import keras
import numpy as np


def load_dataset():
    train_dataset = h5py.File('./Data/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('./Data/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def myModel(input_shape):
    X_input = keras.layers.Input(input_shape)  # Define the input placeholder as a tensor with shape input_shape.

    X = keras.layers.ZeroPadding2D((2, 2))(X_input)  # Zero-Padding
    X = keras.layers.Conv2D(8, (5, 5), strides=(1, 1), name='conv0')(X)  # Convolutional Layer
    X = keras.layers.BatchNormalization(axis=3, name='bn0')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPool2D((2, 2), name='max_pool')(X)  # Pooling Layer

    X = keras.layers.ZeroPadding2D((1, 1))(X)  # Zero-Padding
    X = keras.layers.Conv2D(16, (3, 3), strides=(1, 1), name='conv1')(X)  # Convolutional Layer
    X = keras.layers.BatchNormalization(axis=3, name='bn1')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.AveragePooling2D((2, 2), name='Ave_pool1')(X)  # Pooling Layer

    X = keras.layers.ZeroPadding2D((1, 1))(X)  # Zero-Padding
    X = keras.layers.Conv2D(32, (3, 3), strides=(1, 1), name='conv2')(X)  # Convolutional Layer
    X = keras.layers.BatchNormalization(axis=3, name='bn2')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.AveragePooling2D((2, 2), name='Ave_pool2')(X)  # Pooling Layer

    X = keras.layers.Flatten()(X)  # FLATTEN X (convert it to a vector)
    X = keras.layers.Dense(1, activation='sigmoid', name='fc')(X)  # FULLYCONNECTED

    model = keras.models.Model(inputs=X_input, outputs=X, name='HappyModel')  # Create model.
    return model



# 获取数据，训练集有600张图片，测试集有150张图片，图片的RGB信息存储在X_train_orig和X_test_orig中，
# 每张图片通过RGB这3个通道又分为3个64*64的矩阵存储，即每张图片的维度为（64, 64, 3）。
# Y_train_orig和Y_test_orig是图片的类别标签，值为0和1
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.
# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T


model = myModel((64, 64, 3))  # Create the model
# define a optimizer
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adam)
model.fit(x=X_train, y=Y_train, batch_size=16, epochs=15)
preds = model.evaluate(x=X_test, y=Y_test, batch_size=16)

model.summary()  # Prints a summary of a model
print(preds)

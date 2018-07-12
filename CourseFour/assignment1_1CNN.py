import numpy as np

# Convolutional Neural Networks: Step by Step


# zero pad，4-dimensional matrix only
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    return X_pad


# conv single step
def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)
    return Z


# 卷积层,正向传播
def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C_prev, n_C = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int((n_H_prev + pad * 2 - f) / stride) + 1
    n_W = int((n_W_prev + pad * 2 - f) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)  # zero pad

    for i in range(m):
        ith_A_prev_pad = A_prev_pad[i]
        for H_i in range(n_H):
            for W_i in range(n_W):
                for C_i in range(n_C):
                    # The current "slice"
                    vert_start = H_i * stride
                    vert_end = vert_start + f
                    horiz_start = W_i * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = ith_A_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, H_i, W_i, C_i] = conv_single_step(a_slice_prev, W[:, :, :, C_i], b[:, :, :, C_i])

    cache = (A_prev, W, b, hparameters)
    return Z, cache


# 池化层,正向传播
def pool_forward(A_prev, hparameters, mode="max"):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    A = np.zeros((m, n_H, n_W, n_C_prev))

    for i in range(m):
        ith_A_prev = A_prev[i]
        for H_i in range(n_H):
            for W_i in range(n_W):
                for C_i in range(n_C_prev):
                    vert_start = H_i * stride
                    vert_end = vert_start + f
                    horiz_start = W_i * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = ith_A_prev[vert_start:vert_end, horiz_start:horiz_end, C_i]
                    if mode == "max":
                        A[i, H_i, W_i, C_i] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, H_i, W_i, C_i] = np.mean(a_slice_prev)

    cache = (A_prev, hparameters)
    return A, cache


# 卷积层,反向传播
def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    A_prev, W, b, hparameters = cache
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C_prev, n_C = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    m, n_H, n_W, n_C = dZ.shape

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for H_i in range(n_H):
            for W_i in range(n_W):
                for C_i in range(n_C):
                    # The current "slice"
                    vert_start = H_i * stride
                    vert_end = vert_start + f
                    horiz_start = W_i * stride
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, C_i] * dZ[i, H_i, W_i, C_i]
                    dW[:, :, :, C_i] += a_slice * dZ[i, H_i, W_i, C_i]
                    db[:, :, :, C_i] += dZ[i, H_i, W_i, C_i]

        dA_prev[i, :, :, :] = dA_prev_pad[i, pad:-pad, pad:-pad, :]
        return dA_prev, dW, db


# Creates a mask from an input matrix x, to identify the max entry of x.
def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask


# Distributes the input value in the matrix of dimension shape
def distribute_value(dz, shape):
    n_H, n_W = shape
    average = dz / (n_H * n_W)
    a = np.ones(shape) * average
    return a


# 池化层,反向传播
def pool_backward(dA, cache, mode="max"):
    """
    Implements the backward pass of the pooling layer

    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    A_prev, hparameters = cache
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    stride = hparameters['stride']
    f = hparameters['f']

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))

    for i in range(m):
        a_prev = A_prev[i]

        for H_i in range(n_H):
            for W_i in range(n_W):
                for C_i in range(n_C):
                    vert_start = H_i * stride
                    vert_end = vert_start + f
                    horiz_start = W_i * stride
                    horiz_end = horiz_start + f

                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, C_i]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, C_i] +=\
                            mask * dA[i, vert_start, horiz_start, C_i]
                    elif mode == 'average':
                        da = dA[i, vert_start, horiz_start, C_i]
                        shape = (f, f)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, C_i] += distribute_value(da, shape)

    return dA_prev

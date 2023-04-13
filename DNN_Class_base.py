# -*- coding: utf-8 -*-
"""
Created on 2021.09.08
@author: Xi'an Li
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def pairwise_distance(point_set):
    """Compute pairwise distance of a point cloud.
        Args:
          (x-y)^2 = x^2 - 2xy + y^2
          point_set: tensor (num_points, dims2point)
        Returns:
          pairwise distance: (num_points, num_points)
    """
    point_set_shape = point_set.get_shape().as_list()
    assert(len(point_set_shape)) == 2

    point_set_transpose = tf.transpose(point_set, perm=[1, 0])
    point_set_inner = tf.matmul(point_set, point_set_transpose)
    point_set_inner = -2 * point_set_inner
    point_set_square = tf.reduce_sum(tf.square(point_set), axis=-1, keepdims=True)
    point_set_square_transpose = tf.transpose(point_set_square, perm=[1, 0])
    return point_set_square + point_set_inner + point_set_square_transpose


def np_pairwise_distance(point_set):
    """Compute pairwise distance of a point cloud.
        Args:
          (x-y)^2 = x^2 - 2xy + y^2
          point_set: numpy (num_points, dims2point)
        Returns:
          pairwise distance: (num_points, num_points)
    """
    point_set_shape = point_set.get_shape().as_list()
    assert(len(point_set_shape)) == 2

    point_set_transpose = tf.transpose(point_set, perm=[1, 0])
    point_set_inner = tf.matmul(point_set, point_set_transpose)
    point_set_inner = -2 * point_set_inner
    point_set_square = tf.reduce_sum(tf.square(point_set), axis=-1, keepdims=True)
    point_set_square_transpose = tf.transpose(point_set_square, perm=[1, 0])
    return point_set_square + point_set_inner + point_set_square_transpose


def knn_includeself(dist_matrix, k=20):
    """Get KNN based on the pairwise distance.
        How to use tf.nn.top_k(): https://blog.csdn.net/wuguangbin1230/article/details/72820627
      Args:
        pairwise distance: (num_points, num_points)
        k: int

      Returns:
        nearest neighbors: (num_points, k)
      """
    neg_dist = -1.0*dist_matrix
    _, nn_idx = tf.nn.top_k(neg_dist, k=k)  # 这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
    return nn_idx


def np_knn_includeself(dist_matrix, k=20):
    """Get KNN based on the pairwise distance.
        How to use tf.nn.top_k(): https://blog.csdn.net/wuguangbin1230/article/details/72820627
      Args:
        pairwise distance: (num_points, num_points)
        k: int

      Returns:
        nearest neighbors: (num_points, k)
      """
    neg_dist = -1.0*dist_matrix
    _, nn_idx = np.argpartition(neg_dist, k=k)  # 这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
    return nn_idx


def knn_excludeself(dist_matrix, k=20):
    """Get KNN based on the pairwise distance.
      Args:
        pairwise distance: (num_points, num_points)
        k: int

      Returns:
        nearest neighbors index: (num_points, k)
      """
    neg_dist = -1.0*dist_matrix
    k_neighbors = k+1
    _, knn_idx = tf.nn.top_k(neg_dist, k=k_neighbors)  # 这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
    nn_idx = knn_idx[:, 1: k_neighbors]
    return nn_idx


def get_kneighbors_3D_4DTensor(point_set, nn_idx):
    """Construct neighbors feature for each point
        Args:
        point_set: (batch_size, num_points, 1, dim)
        nn_idx: (batch_size, num_points, k)
        k: int

        Returns:
        neighbors features: (batch_size, num_points, k, dim)
      """
    og_batch_size = point_set.get_shape().as_list()[0]
    og_num_dims = point_set.get_shape().as_list()[-1]
    point_set = tf.squeeze(point_set)
    if og_batch_size == 1:
        point_set = tf.expand_dims(point_set, 0)
    if og_num_dims == 1:
        point_set = tf.expand_dims(point_set, -1)

    point_set_shape = point_set.get_shape()
    batch_size = point_set_shape[0].value
    num_points = point_set_shape[1].value
    num_dims = point_set_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_set_flat = tf.reshape(point_set, [-1, num_dims])
    point_set_neighbors = tf.gather(point_set_flat, nn_idx + idx_)

    return point_set_neighbors


def get_kneighbors_2DTensor(point_set, nn_idx):
    """Construct neighbors feature for each point
        Args:
        point_set: (num_points, dim)
        nn_idx: (num_points, k_num)
        num_points: the number of point
        k_num: the number of neighbor

        Returns:
        neighbors features: (num_points, k_num, dim)
      """
    shape2point_set = point_set.get_shape().as_list()
    assert(len(shape2point_set) == 2)
    point_set_neighbors = tf.gather(point_set, nn_idx)
    return point_set_neighbors


def cal_attends2neighbors(edge_point_set, dis_model='L1'):
    """
        Args:
        edge_point_set:(num_points, k_neighbors, dim2point)
        dis_model:
        return:
        atten_ceof: (num_points, 1, k_neighbors)
    """
    square_edges = tf.square(edge_point_set)                  # (num_points, k_neighbors, dim2point)
    norm2edges = tf.reduce_sum(square_edges, axis=-1)         # (num_points, k_neighbors)
    if str.lower(dis_model) == 'l1':
        norm2edges = tf.sqrt(norm2edges)
    exp_dis = tf.exp(-norm2edges)                             # (num_points, k_neighbors)
    normalize_exp_dis = tf.nn.softmax(exp_dis, axis=-1)
    atten_ceof = tf.expand_dims(normalize_exp_dis, axis=-2)   # (num_points, 1, k_neighbors)
    return atten_ceof


def cal_edgesNorm_attends2neighbors(edge_point_set, dis_model='L1'):
    """
        Args:
        edge_point_set:(num_points, k_neighbors, dim2point)
        dis_model:
        return:
        atten_ceof: (num_points, 1, k_neighbors)
    """
    square_edges = tf.square(edge_point_set)                           # (num_points, k_neighbors, dim2point)
    norm2edges = tf.reduce_sum(square_edges, axis=-1, keepdims=True)   # (num_points, k_neighbors)
    if str.lower(dis_model) == 'l1':
        norm2edges = tf.sqrt(norm2edges)
    normalize_edgeNrom = tf.nn.softmax(norm2edges, axis=1)
    exp_dis = tf.exp(-norm2edges)                                      # (num_points, k_neighbors)
    normalize_exp_dis = tf.nn.softmax(exp_dis, axis=1)
    atten_ceof = tf.transpose(normalize_exp_dis, perm=[0, 2, 1])
    return normalize_edgeNrom, atten_ceof


class np_GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(np_GaussianNormalizer, self).__init__()

        self.mean2x = np.mean(x, axis=-1, keepdims=True)
        self.std2x = np.std(x, axis=-1, keepdims=True)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean2x) / (self.std2x * self.std2x + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std2x * self.std2x + self.eps)) + self.mean2x
        return x

    def get_mean(self):
        return self.mean2x

    def get_std(self):
        return self.std2x


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        shape2x = np.shape(x)
        if len(shape2x) == 3:  # [B, N, D]
            self.mean = np.mean(x, axis=1, keepdims=True)
            self.std = np.std(x, axis=1, keepdims=True)
        elif len(shape2x) == 2:  # [N, D]
            self.mean = np.mean(x, axis=-1, keepdims=True)
            self.std = np.std(x, axis=-1, keepdims=True)
        self.eps = eps

    def encode(self, x):
        # temp = x - self.mean
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std

    def decode_men_std(self, data=None, mean2data=None, std2data=None):
        data = data*(std2data + self.eps) + mean2data
        return data


# ---------------------------------------------- my activations -----------------------------------------------
class my_actFunc(object):
    def __init__(self, actName='linear'):
        super(my_actFunc, self).__init__()
        self.actName = actName

    def __call__(self, x_input):
        if str.lower(self.actName) == 'relu':
            out_x = tf.nn.relu(x_input)
        elif str.lower(self.actName) == 'leaky_relu':
            out_x = tf.nn.leaky_relu(x_input)
        elif str.lower(self.actName) == 'tanh':
            out_x = tf.nn.relu(x_input)
        elif str.lower(self.actName) == 'enhance_tanh' or str.lower(self.actName) == 'enh_tanh':   # 增强的Tanh函数 Enhance Tanh
            out_x = tf.tanh(0.5*np.pi*x_input)
        elif str.lower(self.actName) == 'srelu':
            out_x = tf.nn.relu(x_input)*tf.nn.relu(1-x_input)
        elif str.lower(self.actName) == 's2relu':
            out_x = tf.nn.relu(x_input)*tf.nn.relu(1-x_input)*tf.sin(2*np.pi*x_input)
        elif str.lower(self.actName) == 'elu':
            out_x = tf.nn.elu(x_input)
        elif str.lower(self.actName) == 'gauss':
            out_x = tf.exp(-1.0*x_input*x_input)
        elif str.lower(self.actName) == 'sin':
            out_x = tf.sin(x_input)
        elif str.lower(self.actName) == 'gcu':  # x*cos
            out_x = x_input*tf.cos(x_input)
        elif str.lower(self.actName) == 'sinaddcos':
            out_x = 0.5*(tf.sin(x_input) + tf.cos(x_input))
        elif str.lower(self.actName) == 'sigmoid':
            out_x = tf.nn.sigmoid(x_input)
        elif str.lower(self.actName) == 'softplus':
            out_x = tf.nn.softplus(x_input)
        elif str.lower(self.actName) == 'mish':
            out_x = x_input*tf.tanh(tf.math.log(1+tf.exp(x_input)))
        elif str.lower(self.actName) == 'fourier':
            out_x = tf.concat([tf.sin(x_input), tf.cos(x_input)], axis=-1)
        elif str.lower(self.actName) == 'gelu':
            # temp2x = np.sqrt(2 / np.pi) * (x_input + 0.044715 * x_input * x_input * x_input)
            # out_x = 0.5 * x_input + 0.5 * x_input * tf.tanh(temp2x)    # 原来的gelu输出
            out_x = x_input*tf.exp(x_input)/(1+tf.exp(x_input))
        elif str.lower(self.actName) == 'mgelu':
            temp2x = np.sqrt(2 / np.pi) * (x_input + 0.044715 * x_input * x_input * x_input)
            # out_x = 0.5 * x_input + 0.5 * x_input * tf.tanh(temp2x)  # 原来的gelu输出
            # out_x = 0.5 * + 0.5 * x_input * tf.tanh(temp2x)          # 我写错的gelu输出
            out_x = 0.25 * x_input * tf.tanh(temp2x)                   # 我的Gelu 输出(误打误撞，得到一个好的激活函数, 效果最好)
            # out_x = 0.25 * x_input * tf.tanh(temp2x) + 0.25
            # out_x = 0.5*x_input * tf.tanh(temp2x)
            # out_x = x_input * tf.tanh(temp2x)
        else:
            out_x = x_input
        return out_x

    # The following activation functions are previous works in the old version of DNN_base
    def sinAddcos(self, x):
        return 0.5*(tf.sin(x) + tf.cos(x))
        # return tf.sin(x) + tf.cos(x)

    def sinAddcos_sReLu(self, x):
        return tf.nn.relu(1-x)*tf.nn.relu(x)*(tf.sin(2*np.pi*x) + tf.cos(2*np.pi*x))

    def s3relu(self, x):
        # return 0.5*tf.nn.relu(1-x)*tf.nn.relu(1+x)*tf.sin(2*np.pi*x)
        # return 0.21*tf.nn.relu(1-x)*tf.nn.relu(1+x)*tf.sin(2*np.pi*x)
        # return tf.nn.relu(1 - x) * tf.nn.relu(x) * (tf.sin(2 * np.pi * x) + tf.cos(2 * np.pi * x))   # (work不好)
        # return tf.nn.relu(1 - x) * tf.nn.relu(1 + x) * (tf.sin(2 * np.pi * x) + tf.cos(2 * np.pi * x)) #（不work）
        return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(2*np.pi*tf.abs(x))      # work 不如 s2relu
        # return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(2*np.pi*x)            # work 不如 s2relu
        # return 1.5*tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(np.pi*x)
        # return tf.nn.relu(1 - x) * tf.nn.relu(x+0.5) * tf.sin(2 * np.pi * x)

    def csrelu(self, x):
        # return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.cos(np.pi*x)
        return 1.5*tf.nn.relu(1 - x) * tf.nn.relu(x) * tf.cos(np.pi * x)
        # return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.cos(np.pi*x)

    def stanh(self, x):
        # return tf.tanh(x)*tf.sin(2*np.pi*x)
        return tf.sin(2*np.pi*tf.tanh(x))

    def mexican(self, x):
        return (1-x*x)*tf.exp(-0.5*x*x)

    def modify_mexican(self, x):
        # return 1.25*x*tf.exp(-0.25*x*x)
        # return x * tf.exp(-0.125 * x * x)
        return x * tf.exp(-0.075*x * x)
        # return -1.25*x*tf.exp(-0.25*x*x)

    def sm_mexican(self, x):
        # return tf.sin(np.pi*x) * x * tf.exp(-0.075*x * x)
        # return tf.sin(np.pi*x) * x * tf.exp(-0.125*x * x)
        return 2.0*tf.sin(np.pi*x) * x * tf.exp(-0.5*x * x)

    def singauss(self, x):
        # return 0.6 * tf.exp(-4 * x * x) * tf.sin(np.pi * x)
        # return 0.6 * tf.exp(-5 * x * x) * tf.sin(np.pi * x)
        # return 0.75*tf.exp(-5*x*x)*tf.sin(2*np.pi*x)
        # return tf.exp(-(x-0.5) * (x - 0.5)) * tf.sin(np.pi * x)
        # return 0.25 * tf.exp(-3.5 * x * x) * tf.sin(2 * np.pi * x)
        # return 0.225*tf.exp(-2.5 * (x - 0.5) * (x - 0.5)) * tf.sin(2*np.pi * x)
        return 0.225 * tf.exp(-2 * (x - 0.5) * (x - 0.5)) * tf.sin(2 * np.pi * x)
        # return 0.4 * tf.exp(-10 * (x - 0.5) * (x - 0.5)) * tf.sin(2 * np.pi * x)
        # return 0.45 * tf.exp(-5 * (x - 1.0) * (x - 1.0)) * tf.sin(np.pi * x)
        # return 0.3 * tf.exp(-5 * (x - 1.0) * (x - 1.0)) * tf.sin(2 * np.pi * x)
        # return tf.sin(2*np.pi*tf.exp(-0.5*x*x))

    def powsin_srelu(self, x):
        return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)*tf.sin(2*np.pi*x)

    def sin2_srelu(self, x):
        return 2.0*tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(4*np.pi*x)*tf.sin(2*np.pi*x)

    def slrelu(self, x):
        return tf.nn.leaky_relu(1-x)*tf.nn.leaky_relu(x)

    def pow2relu(self, x):
        return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.nn.relu(x)

    def selu(self, x):
        return tf.nn.elu(1-x)*tf.nn.elu(x)

    def wave(self, x):
        return tf.nn.relu(x) - 2*tf.nn.relu(x-1/4) + \
               2*tf.nn.relu(x-3/4) - tf.nn.relu(x-1)

    def phi(self, x):
        return tf.nn.relu(x) * tf.nn.relu(x)-3*tf.nn.relu(x-1)*tf.nn.relu(x-1) + 3*tf.nn.relu(x-2)*tf.nn.relu(x-2) \
               - tf.nn.relu(x-3)*tf.nn.relu(x-3)*tf.nn.relu(x-3)


# This class of Dense_Net is an union for the normal-DNN, Scale-DNN and Fourier-DNN
class DenseNet(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5):
        super(DenseNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName2in = actName2in
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float

        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        self.Ws = []
        self.Bs = []
        with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
            if str.lower(self.name2Model) == 'fourier_dnn':
                stddev_WB = (2.0 / (indim + hidden_units[0])) ** varcoe
                Win = tf.compat.v1.get_variable(
                    name=str(scope2W) + '_in', shape=(indim, hidden_units[0]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
                Bin = tf.compat.v1.get_variable(
                    name=str(scope2B) + '_in', shape=(hidden_units[0],),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=self.float_type, trainable=False)
                self.Ws.append(Win)
                self.Bs.append(Bin)
                for i_layer in range(len(hidden_units)-1):
                    stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                    if i_layer == 0:
                        W = tf.compat.v1.get_variable(
                            name=str(scope2W)+str(i_layer), shape=(hidden_units[i_layer]*2, hidden_units[i_layer + 1]),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                            dtype=self.float_type)
                        B = tf.compat.v1.get_variable(
                            name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                            dtype=self.float_type)
                    else:
                        W = tf.compat.v1.get_variable(
                            name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                            dtype=self.float_type)
                        B = tf.compat.v1.get_variable(
                            name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                            dtype=self.float_type)
                    self.Ws.append(W)
                    self.Bs.append(B)
            else:
                stddev_WB = (2.0 / (indim + hidden_units[0])) ** varcoe
                Win = tf.compat.v1.get_variable(
                    name=str(scope2W) + '_in', shape=(indim, hidden_units[0]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
                Bin = tf.compat.v1.get_variable(
                    name=str(scope2B) + '_in', shape=(hidden_units[0],),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=self.float_type, trainable=True)
                self.Ws.append(Win)
                self.Bs.append(Bin)
                for i_layer in range(len(hidden_units) - 1):
                    stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** 0.5
                    W = tf.compat.v1.get_variable(
                        name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                        dtype=self.float_type)
                    B = tf.compat.v1.get_variable(
                        name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                        dtype=self.float_type)
                    self.Ws.append(W)
                    self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.compat.v1.get_variable(
                name=str(scope2W) + '_out', shape=(hidden_units[-1], outdim),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
            Bout = tf.compat.v1.get_variable(
                name=str(scope2B) + '_out', shape=(outdim,), initializer=tf.random_normal_initializer(stddev=stddev_WB),
                trainable=True, dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=0.5):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H = tf.add(tf.matmul(inputs, self.Ws[0]), self.Bs[0])
        if str.lower(self.name2Model) == 'dnn':  # name2Model:DNN
            H = self.actFunc_in(H)
        else:
            assert (len(scale) != 0)
            repeat_num = int(self.hidden_units[0] / len(scale))
            repeat_scale = np.repeat(scale, repeat_num)

            if self.repeat_high_freq:
                repeat_scale = np.concatenate(
                    (repeat_scale, np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[-1]))
            else:
                repeat_scale = np.concatenate(
                    (np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[0], repeat_scale))

            if self.type2float == 'float32':
                repeat_scale = repeat_scale.astype(np.float32)
            elif self.type2float == 'float64':
                repeat_scale = repeat_scale.astype(np.float64)
            else:
                repeat_scale = repeat_scale.astype(np.float16)

            if str.lower(self.name2Model) == 'fourier_dnn':
                H = sFourier * self.actFunc_in(H * repeat_scale)
            else:
                H = self.actFunc_in(H * repeat_scale)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H_pre = H
            H = tf.add(tf.matmul(H, self.Ws[i_layer + 1]), self.Bs[i_layer + 1])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


# This class of Dense_Net is an union for the normal-DNN, Scale-DNN and Fourier-DNN, but it have some problems.
# I implement them by three classes:Pure_Dense_Net, Dense_ScaleNet, Dense_Fourier_Net
class Dense_Net(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5):
        super(Dense_Net, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName2in = actName2in
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float

        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        self.Ws = []
        self.Bs = []
        with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
            if str.lower(self.name2Model) == 'fourier_dnn':
                stddev_WB = (2.0 / (indim + hidden_units[0])) ** varcoe
                Win = tf.compat.v1.get_variable(
                    name=str(scope2W) + '_in', shape=(indim, hidden_units[0]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
                Bin = tf.compat.v1.get_variable(
                    name=str(scope2B) + '_in', shape=(hidden_units[0],),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=self.float_type, trainable=False)
                self.Ws.append(Win)
                self.Bs.append(Bin)
                for i_layer in range(len(hidden_units)-1):
                    stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                    if i_layer == 0:
                        W = tf.compat.v1.get_variable(
                            name=str(scope2W)+str(i_layer), shape=(hidden_units[i_layer]*2, hidden_units[i_layer + 1]),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                            dtype=self.float_type)
                        B = tf.compat.v1.get_variable(
                            name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                            dtype=self.float_type)
                    else:
                        W = tf.compat.v1.get_variable(
                            name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                            dtype=self.float_type)
                        B = tf.compat.v1.get_variable(
                            name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                            dtype=self.float_type)
                    self.Ws.append(W)
                    self.Bs.append(B)
            else:
                stddev_WB = (2.0 / (indim + hidden_units[0])) ** varcoe
                Win = tf.compat.v1.get_variable(
                    name=str(scope2W) + '_in', shape=(indim, hidden_units[0]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
                Bin = tf.compat.v1.get_variable(
                    name=str(scope2B) + '_in', shape=(hidden_units[0],),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=self.float_type, trainable=True)
                self.Ws.append(Win)
                self.Bs.append(Bin)
                for i_layer in range(len(hidden_units) - 1):
                    stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** 0.5
                    W = tf.compat.v1.get_variable(
                        name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                        dtype=self.float_type)
                    B = tf.compat.v1.get_variable(
                        name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                        dtype=self.float_type)
                    self.Ws.append(W)
                    self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.compat.v1.get_variable(
                name=str(scope2W) + '_out', shape=(hidden_units[-1], outdim),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
            Bout = tf.compat.v1.get_variable(
                name=str(scope2B) + '_out', shape=(outdim,), initializer=tf.random_normal_initializer(stddev=stddev_WB),
                trainable=True, dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H = tf.add(tf.matmul(inputs, self.Ws[0]), self.Bs[0])
        if str.lower(self.name2Model) == 'fourier_dnn':
            assert (len(scale) != 0)
            repeat_num = int(self.hidden_units[0] / len(scale))
            repeat_scale = np.repeat(scale, repeat_num)

            if self.repeat_high_freq:
                repeat_scale = np.concatenate(
                    (repeat_scale, np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[-1]))
            else:
                repeat_scale = np.concatenate(
                    (np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[0], repeat_scale))

            if self.type2float == 'float32':
                repeat_scale = repeat_scale.astype(np.float32)
            elif self.type2float == 'float64':
                repeat_scale = repeat_scale.astype(np.float64)
            else:
                repeat_scale = repeat_scale.astype(np.float16)

            if str.lower(self.actName) == 's2relu':
                H = 0.5 * tf.concat([tf.cos(H * repeat_scale), tf.sin(H * repeat_scale)], axis=-1)
            else:
                H = tf.concat([tf.cos(H * repeat_scale), tf.sin(H * repeat_scale)], axis=-1)
        elif str.lower(self.name2Model) == 'scale_dnn' or str.lower(self.name2Model) == 'wavelet_dnn':
            assert (len(scale) != 0)
            repeat_num = int(self.hidden_units[0] / len(scale))
            repeat_scale = np.repeat(scale, repeat_num)

            if self.repeat_high_freq:
                repeat_scale = np.concatenate(
                    (repeat_scale, np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[-1]))
            else:
                repeat_scale = np.concatenate(
                    (np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[0], repeat_scale))

            if self.type2float == 'float32':
                repeat_scale = repeat_scale.astype(np.float32)
            elif self.type2float == 'float64':
                repeat_scale = repeat_scale.astype(np.float64)
            else:
                repeat_scale = repeat_scale.astype(np.float16)

            H = self.actFunc_in(H * repeat_scale)
        else:  # name2Model:DNN, RBF_DNN
            H = self.actFunc_in(H)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H_pre = H
            H = tf.add(tf.matmul(H, self.Ws[i_layer + 1]), self.Bs[i_layer + 1])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Pure_Dense_Net(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', type2float='float32', varcoe=0.5):
        super(Pure_Dense_Net, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName2in = actName2in
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.type2float = type2float
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
            stddev_WB = (2.0 / (indim + hidden_units[0])) ** varcoe
            Win = tf.compat.v1.get_variable(name=str(scope2W) + '_in', shape=(indim, hidden_units[0]),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                            dtype=self.float_type)
            Bin = tf.compat.v1.get_variable(name=str(scope2B) + '_in', shape=(hidden_units[0],),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                            dtype=self.float_type, trainable=True)
            self.Ws.append(Win)
            self.Bs.append(Bin)
            for i_layer in range(len(hidden_units) - 1):
                stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                W = tf.compat.v1.get_variable(
                    name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                B = tf.compat.v1.get_variable(
                    name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                self.Ws.append(W)
                self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.compat.v1.get_variable(
                name=str(scope2W) + '_out', shape=(hidden_units[-1], outdim),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
            Bout = tf.compat.v1.get_variable(
                name=str(scope2B) + '_out', shape=(outdim,),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=1.0):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data, it is not useful in this Pure_Dense_Net
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H = tf.add(tf.matmul(inputs, self.Ws[0]), self.Bs[0])
        H = self.actFunc_in(H)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H_pre = H
            H = tf.add(tf.matmul(H, self.Ws[i_layer + 1]), self.Bs[i_layer + 1])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Dense_ScaleNet(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5):
        super(Dense_ScaleNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName2in = actName2in
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
            stddev_WB = (2.0 / (indim + hidden_units[0])) ** varcoe
            Win = tf.compat.v1.get_variable(name=str(scope2W) + '_in', shape=(indim, hidden_units[0]),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                                            dtype=self.float_type)
            Bin = tf.compat.v1.get_variable(name=str(scope2B) + '_in', shape=(hidden_units[0],),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                            dtype=self.float_type, trainable=True)
            self.Ws.append(Win)
            self.Bs.append(Bin)
            for i_layer in range(len(hidden_units) - 1):
                stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                W = tf.compat.v1.get_variable(
                    name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
                B = tf.compat.v1.get_variable(
                    name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
                self.Ws.append(W)
                self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.compat.v1.get_variable(
                name=str(scope2W) + '_out', shape=(hidden_units[-1], outdim),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
            Bout = tf.compat.v1.get_variable(
                name=str(scope2B) + '_out', shape=(outdim,), initializer=tf.random_normal_initializer(stddev=stddev_WB),
                trainable=True, dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=1.0):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H = tf.add(tf.matmul(inputs, self.Ws[0]), self.Bs[0])
        assert (len(scale) != 0)
        repeat_num = int(self.hidden_units[0] / len(scale))
        repeat_scale = np.repeat(scale, repeat_num)

        if self.repeat_high_freq:
            repeat_scale = np.concatenate(
                (repeat_scale, np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[-1]))
        else:
            repeat_scale = np.concatenate(
                (np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[0], repeat_scale))

        if self.type2float == 'float32':
            repeat_scale = repeat_scale.astype(np.float32)
        elif self.type2float == 'float64':
            repeat_scale = repeat_scale.astype(np.float64)
        else:
            repeat_scale = repeat_scale.astype(np.float16)

        H = self.actFunc_in(H * repeat_scale)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H_pre = H
            H = tf.add(tf.matmul(H, self.Ws[i_layer + 1]), self.Bs[i_layer + 1])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Dense_FourierNet(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5):
        super(Dense_FourierNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
            stddev_WB = (2.0 / (indim + hidden_units[0])) ** varcoe
            Win = tf.compat.v1.get_variable(
                name=str(scope2W) + '_in', shape=(indim, hidden_units[0]),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
            # Bin = tf.compat.v1.get_variable(
            #     name=str(scope2B) + '_in', shape=(hidden_units[0],),
            #     initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=self.float_type, trainable=False)
            Bin = tf.compat.v1.get_variable(
                name=str(scope2B) + '_in', shape=(hidden_units[0],),
                initializer=tf.zeros_initializer(), dtype=self.float_type, trainable=False)
            self.Ws.append(Win)
            self.Bs.append(Bin)
            for i_layer in range(len(hidden_units)-1):
                stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                if i_layer == 0:
                    W = tf.compat.v1.get_variable(
                        name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer] * 2, hidden_units[i_layer + 1]),
                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                        dtype=self.float_type)
                    B = tf.compat.v1.get_variable(
                        name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                        dtype=self.float_type)
                else:
                    W = tf.compat.v1.get_variable(
                        name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                        dtype=self.float_type)
                    B = tf.compat.v1.get_variable(
                        name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                        initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                        dtype=self.float_type)
                self.Ws.append(W)
                self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.compat.v1.get_variable(
                name=str(scope2W) + '_out', shape=(hidden_units[-1], outdim),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
            Bout = tf.compat.v1.get_variable(
                name=str(scope2B) + '_out', shape=(outdim,), initializer=tf.random_normal_initializer(stddev=stddev_WB),
                trainable=True, dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=1.0):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H = tf.add(tf.matmul(inputs, self.Ws[0]), self.Bs[0])
        assert (len(scale) != 0)
        repeat_num = int(self.hidden_units[0] / len(scale))
        repeat_scale = np.repeat(scale, repeat_num)

        if self.repeat_high_freq:
            repeat_scale = np.concatenate(
                (repeat_scale, np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[-1]))
        else:
            repeat_scale = np.concatenate(
                (np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[0], repeat_scale))

        if self.type2float == 'float32':
            repeat_scale = repeat_scale.astype(np.float32)
        elif self.type2float == 'float64':
            repeat_scale = repeat_scale.astype(np.float64)
        else:
            repeat_scale = repeat_scale.astype(np.float16)

        H = sFourier*tf.concat([tf.cos(H * repeat_scale), tf.sin(H * repeat_scale)], axis=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H_pre = H
            H = tf.add(tf.matmul(H, self.Ws[i_layer + 1]), self.Bs[i_layer + 1])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record and i_layer != 0:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Fourier_FeatureDNN(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5, sigma=10.0, trainable2sigma=False):
        super(Fourier_FeatureDNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.sigma = sigma
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
            self.W_FF = tf.compat.v1.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma,
                                              dtype=self.float_type, trainable=trainable2sigma,
                                              name=str(scope2W) + 'FF')

            for i_layer in range(len(hidden_units)-1):
                stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                W = tf.compat.v1.get_variable(
                    name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                B = tf.compat.v1.get_variable(
                    name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                self.Ws.append(W)
                self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.compat.v1.get_variable(
                name=str(scope2W) + '_out', shape=(hidden_units[-1], outdim),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
            Bout = tf.compat.v1.get_variable(
                name=str(scope2B) + '_out', shape=(outdim,), initializer=tf.random_normal_initializer(stddev=stddev_WB),
                trainable=True, dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=1.0):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H_FF = tf.matmul(inputs, self.W_FF)

        H = sFourier*tf.concat([tf.cos(H_FF), tf.sin(H_FF)], axis=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H_pre = H
            # WW = self.Ws[i_layer]
            # BB = self.Bs[i_layer]
            H = tf.add(tf.matmul(H, self.Ws[i_layer]), self.Bs[i_layer])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record and i_layer != 0:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Multi_2FF_DNN(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5, sigma1=10.0, sigma2=10.0, trainable2sigma=False):
        super(Multi_2FF_DNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
            self.W_FF1 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma1,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF1')

            self.W_FF2 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma2,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF2')

            for i_layer in range(len(hidden_units)-1):
                stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                W = tf.compat.v1.get_variable(
                    name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                B = tf.compat.v1.get_variable(
                    name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                self.Ws.append(W)
                self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.compat.v1.get_variable(
                name=str(scope2W) + '_out', shape=(2*hidden_units[-1], outdim),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
            Bout = tf.compat.v1.get_variable(
                name=str(scope2B) + '_out', shape=(outdim,), initializer=tf.random_normal_initializer(stddev=stddev_WB),
                trainable=True, dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=1.0):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H_FF1 = tf.matmul(inputs, self.W_FF1)
        H_FF2 = tf.matmul(inputs, self.W_FF2)

        H1 = sFourier*tf.concat([tf.cos(H_FF1), tf.sin(H_FF1)], axis=-1)
        H2 = sFourier * tf.concat([tf.cos(H_FF2), tf.sin(H_FF2)], axis=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H1_pre = H1
            H1 = tf.add(tf.matmul(H1, self.Ws[i_layer]), self.Bs[i_layer])
            H1 = self.actFunc(H1)

            H2_pre = H2
            H2 = tf.add(tf.matmul(H2, self.Ws[i_layer]), self.Bs[i_layer])
            H2 = self.actFunc(H2)
            if self.hidden_units[i_layer + 1] == hidden_record and i_layer != 0:
                H1 = H1 + H1_pre
                H2 = H2 + H2_pre

            hidden_record = self.hidden_units[i_layer + 1]
        H_concat = tf.concat([H1, H2], axis=-1)
        H = tf.add(tf.matmul(H_concat, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Multi_3FF_DNN(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5, sigma1=10.0, sigma2=10.0, sigma3=50.0, trainable2sigma=False):
        super(Multi_3FF_DNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
            self.W_FF1 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma1,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF1')

            self.W_FF2 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma2,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF2')

            self.W_FF3 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma3,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF3')

            for i_layer in range(len(hidden_units)-1):
                stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                W = tf.compat.v1.get_variable(
                    name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                B = tf.compat.v1.get_variable(
                    name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                self.Ws.append(W)
                self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.compat.v1.get_variable(
                name=str(scope2W) + '_out', shape=(3*hidden_units[-1], outdim),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
            Bout = tf.compat.v1.get_variable(
                name=str(scope2B) + '_out', shape=(outdim,), initializer=tf.random_normal_initializer(stddev=stddev_WB),
                trainable=True, dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=1.0):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H_FF1 = tf.matmul(inputs, self.W_FF1)
        H_FF2 = tf.matmul(inputs, self.W_FF2)
        H_FF3 = tf.matmul(inputs, self.W_FF3)

        H1 = sFourier*tf.concat([tf.cos(H_FF1), tf.sin(H_FF1)], axis=-1)
        H2 = sFourier * tf.concat([tf.cos(H_FF2), tf.sin(H_FF2)], axis=-1)
        H3 = sFourier * tf.concat([tf.cos(H_FF3), tf.sin(H_FF3)], axis=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H1_pre = H1
            H1 = tf.add(tf.matmul(H1, self.Ws[i_layer]), self.Bs[i_layer])
            H1 = self.actFunc(H1)

            H2_pre = H2
            H2 = tf.add(tf.matmul(H2, self.Ws[i_layer]), self.Bs[i_layer])
            H2 = self.actFunc(H2)

            H3_pre = H3
            H3 = tf.add(tf.matmul(H3, self.Ws[i_layer]), self.Bs[i_layer])
            H3 = self.actFunc(H3)
            if self.hidden_units[i_layer + 1] == hidden_record and i_layer != 0:
                H1 = H1 + H1_pre
                H2 = H2 + H2_pre
                H3 = H3 + H3_pre

            hidden_record = self.hidden_units[i_layer + 1]
        H_concat = tf.concat([H1, H2, H3], axis=-1)
        H = tf.add(tf.matmul(H_concat, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Multi_4FF_DNN(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """

    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5, sigma1=10.0, sigma2=10.0, sigma3=50.0, sigma4=100.0, trainable2sigma=False):
        super(Multi_4FF_DNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.sigma4 = sigma4
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
            self.W_FF1 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma1,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF1')

            self.W_FF2 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma2,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF2')

            self.W_FF3 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma3,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF3')

            self.W_FF4 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma4,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF4')

            for i_layer in range(len(hidden_units) - 1):
                stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                W = tf.compat.v1.get_variable(
                    name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                B = tf.compat.v1.get_variable(
                    name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                self.Ws.append(W)
                self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.compat.v1.get_variable(
                name=str(scope2W) + '_out', shape=(4 * hidden_units[-1], outdim),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
            Bout = tf.compat.v1.get_variable(
                name=str(scope2B) + '_out', shape=(outdim,), initializer=tf.random_normal_initializer(stddev=stddev_WB),
                trainable=True, dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=1.0):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H_FF1 = tf.matmul(inputs, self.W_FF1)
        H_FF2 = tf.matmul(inputs, self.W_FF2)
        H_FF3 = tf.matmul(inputs, self.W_FF3)
        H_FF4 = tf.matmul(inputs, self.W_FF4)

        H1 = sFourier * tf.concat([tf.cos(H_FF1), tf.sin(H_FF1)], axis=-1)
        H2 = sFourier * tf.concat([tf.cos(H_FF2), tf.sin(H_FF2)], axis=-1)
        H3 = sFourier * tf.concat([tf.cos(H_FF3), tf.sin(H_FF3)], axis=-1)
        H4 = sFourier * tf.concat([tf.cos(H_FF4), tf.sin(H_FF4)], axis=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H1_pre = H1
            H1 = tf.add(tf.matmul(H1, self.Ws[i_layer]), self.Bs[i_layer])
            H1 = self.actFunc(H1)

            H2_pre = H2
            H2 = tf.add(tf.matmul(H2, self.Ws[i_layer]), self.Bs[i_layer])
            H2 = self.actFunc(H2)

            H3_pre = H3
            H3 = tf.add(tf.matmul(H3, self.Ws[i_layer]), self.Bs[i_layer])
            H3 = self.actFunc(H3)

            H4_pre = H4
            H4 = tf.add(tf.matmul(H4, self.Ws[i_layer]), self.Bs[i_layer])
            H4 = self.actFunc(H4)
            if self.hidden_units[i_layer + 1] == hidden_record and i_layer != 0:
                H1 = H1 + H1_pre
                H2 = H2 + H2_pre
                H3 = H3 + H3_pre
                H4 = H4 + H4_pre

            hidden_record = self.hidden_units[i_layer + 1]
        H_concat = tf.concat([H1, H2, H3, H4], axis=-1)
        H = tf.add(tf.matmul(H_concat, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Multi_5FF_DNN(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """

    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5, sigma1=10.0, sigma2=10.0, sigma3=20.0, sigma4=50.0, sigma5=100.0, trainable2sigma=False):
        super(Multi_5FF_DNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.sigma4 = sigma4
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
            self.W_FF1 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma1,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF1')

            self.W_FF2 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma2,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF2')

            self.W_FF3 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma3,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF3')

            self.W_FF4 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma4,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF4')

            self.W_FF5 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma5,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF5')

            for i_layer in range(len(hidden_units) - 1):
                stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                W = tf.compat.v1.get_variable(
                    name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                B = tf.compat.v1.get_variable(
                    name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                self.Ws.append(W)
                self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.compat.v1.get_variable(
                name=str(scope2W) + '_out', shape=(5 * hidden_units[-1], outdim),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
            Bout = tf.compat.v1.get_variable(
                name=str(scope2B) + '_out', shape=(outdim,), initializer=tf.random_normal_initializer(stddev=stddev_WB),
                trainable=True, dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=1.0):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H_FF1 = tf.matmul(inputs, self.W_FF1)
        H_FF2 = tf.matmul(inputs, self.W_FF2)
        H_FF3 = tf.matmul(inputs, self.W_FF3)
        H_FF4 = tf.matmul(inputs, self.W_FF4)
        H_FF5 = tf.matmul(inputs, self.W_FF5)

        H1 = sFourier * tf.concat([tf.cos(H_FF1), tf.sin(H_FF1)], axis=-1)
        H2 = sFourier * tf.concat([tf.cos(H_FF2), tf.sin(H_FF2)], axis=-1)
        H3 = sFourier * tf.concat([tf.cos(H_FF3), tf.sin(H_FF3)], axis=-1)
        H4 = sFourier * tf.concat([tf.cos(H_FF4), tf.sin(H_FF4)], axis=-1)
        H5 = sFourier * tf.concat([tf.cos(H_FF5), tf.sin(H_FF5)], axis=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H1_pre = H1
            H2_pre = H2
            H3_pre = H3
            H4_pre = H4
            H5_pre = H5

            H1 = tf.add(tf.matmul(H1, self.Ws[i_layer]), self.Bs[i_layer])
            H1 = self.actFunc(H1)

            H2 = tf.add(tf.matmul(H2, self.Ws[i_layer]), self.Bs[i_layer])
            H2 = self.actFunc(H2)

            H3 = tf.add(tf.matmul(H3, self.Ws[i_layer]), self.Bs[i_layer])
            H3 = self.actFunc(H3)

            H4 = tf.add(tf.matmul(H4, self.Ws[i_layer]), self.Bs[i_layer])
            H4 = self.actFunc(H4)

            H5 = tf.add(tf.matmul(H5, self.Ws[i_layer]), self.Bs[i_layer])
            H5 = self.actFunc(H5)
            if self.hidden_units[i_layer + 1] == hidden_record and i_layer != 0:
                H1 = H1 + H1_pre
                H2 = H2 + H2_pre
                H3 = H3 + H3_pre
                H4 = H4 + H4_pre
                H5 = H5 + H5_pre

            hidden_record = self.hidden_units[i_layer + 1]
        H_concat = tf.concat([H1, H2, H3, H4, H5], axis=-1)
        H = tf.add(tf.matmul(H_concat, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Multi_8FF_DNN(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """

    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5, sigma1=1.0, sigma2=5.0, sigma3=10.0, sigma4=20.0, sigma5=40.0, sigma6=60.0, sigma7=80.0,
                 sigma8=100.0, trainable2sigma=False):
        super(Multi_8FF_DNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.sigma4 = sigma4
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
            # self.W_FF = tf.compat.v1.get_variable(
            #     name=str(scope2W) + 'FF', shape=(indim, hidden_units[0]//2),
            #     initializer=tf.random_normal_initializer(stddev=float(sigma)), trainable=False,
            #     dtype=self.float_type)
            self.W_FF1 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma1,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF1')

            self.W_FF2 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma2,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF2')

            self.W_FF3 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma3,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF3')

            self.W_FF4 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma4,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF4')

            self.W_FF5 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma5,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF5')

            self.W_FF6 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma6,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF6')

            self.W_FF7 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma7,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'F7')

            self.W_FF8 = tf.Variable(tf.random_normal([indim, hidden_units[0] // 2], dtype=tf.float32) * sigma8,
                                     dtype=self.float_type, trainable=trainable2sigma, name=str(scope2W) + 'FF8')

            for i_layer in range(len(hidden_units) - 1):
                stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                W = tf.compat.v1.get_variable(
                    name=str(scope2W) + str(i_layer), shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                B = tf.compat.v1.get_variable(
                    name=str(scope2B) + str(i_layer), shape=(hidden_units[i_layer + 1],),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True,
                    dtype=self.float_type)
                self.Ws.append(W)
                self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.compat.v1.get_variable(
                name=str(scope2W) + '_out', shape=(8 * hidden_units[-1], outdim),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), trainable=True, dtype=self.float_type)
            Bout = tf.compat.v1.get_variable(
                name=str(scope2B) + '_out', shape=(outdim,), initializer=tf.random_normal_initializer(stddev=stddev_WB),
                trainable=True, dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=1.0):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H_FF1 = tf.matmul(inputs, self.W_FF1)
        H_FF2 = tf.matmul(inputs, self.W_FF2)
        H_FF3 = tf.matmul(inputs, self.W_FF3)
        H_FF4 = tf.matmul(inputs, self.W_FF4)
        H_FF5 = tf.matmul(inputs, self.W_FF5)
        H_FF6 = tf.matmul(inputs, self.W_FF6)
        H_FF7 = tf.matmul(inputs, self.W_FF7)
        H_FF8 = tf.matmul(inputs, self.W_FF8)

        H1 = sFourier * tf.concat([tf.cos(H_FF1), tf.sin(H_FF1)], axis=-1)
        H2 = sFourier * tf.concat([tf.cos(H_FF2), tf.sin(H_FF2)], axis=-1)
        H3 = sFourier * tf.concat([tf.cos(H_FF3), tf.sin(H_FF3)], axis=-1)
        H4 = sFourier * tf.concat([tf.cos(H_FF4), tf.sin(H_FF4)], axis=-1)
        H5 = sFourier * tf.concat([tf.cos(H_FF5), tf.sin(H_FF5)], axis=-1)
        H6 = sFourier * tf.concat([tf.cos(H_FF6), tf.sin(H_FF6)], axis=-1)
        H7 = sFourier * tf.concat([tf.cos(H_FF7), tf.sin(H_FF7)], axis=-1)
        H8 = sFourier * tf.concat([tf.cos(H_FF8), tf.sin(H_FF8)], axis=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units) - 1):
            H1_pre = H1
            H2_pre = H2
            H3_pre = H3
            H4_pre = H4
            H5_pre = H5
            H6_pre = H6
            H7_pre = H7
            H8_pre = H8

            H1 = tf.add(tf.matmul(H1, self.Ws[i_layer]), self.Bs[i_layer])
            H1 = self.actFunc(H1)

            H2 = tf.add(tf.matmul(H2, self.Ws[i_layer]), self.Bs[i_layer])
            H2 = self.actFunc(H2)

            H3 = tf.add(tf.matmul(H3, self.Ws[i_layer]), self.Bs[i_layer])
            H3 = self.actFunc(H3)

            H4 = tf.add(tf.matmul(H4, self.Ws[i_layer]), self.Bs[i_layer])
            H4 = self.actFunc(H4)

            H5 = tf.add(tf.matmul(H5, self.Ws[i_layer]), self.Bs[i_layer])
            H5 = self.actFunc(H5)

            H6 = tf.add(tf.matmul(H6, self.Ws[i_layer]), self.Bs[i_layer])
            H6 = self.actFunc(H6)

            H7 = tf.add(tf.matmul(H7, self.Ws[i_layer]), self.Bs[i_layer])
            H7 = self.actFunc(H7)

            H8 = tf.add(tf.matmul(H8, self.Ws[i_layer]), self.Bs[i_layer])
            H8 = self.actFunc(H8)
            if self.hidden_units[i_layer + 1] == hidden_record and i_layer != 0:
                H1 = H1 + H1_pre
                H2 = H2 + H2_pre
                H3 = H3 + H3_pre
                H4 = H4 + H4_pre
                H5 = H5 + H5_pre
                H6 = H6 + H6_pre
                H7 = H7 + H7_pre
                H8 = H8 + H8_pre

            hidden_record = self.hidden_units[i_layer + 1]
        H_concat = tf.concat([H1, H2, H3, H4, H5, H6, H7, H8], axis=-1)
        H = tf.add(tf.matmul(H_concat, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Scale_SubNets(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        num2subnets: the number of sub-nets
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5, num2subnets=5):
        super(Scale_SubNets, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName2in = actName2in
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.num2subnets = num2subnets

        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        self.Ws = []
        self.Bs = []
        for i_net in range(num2subnets):
            with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
                stddev_WB = (2.0 / (indim + hidden_units[0])) ** varcoe
                Win = tf.compat.v1.get_variable(name=str(scope2W) + str(i_net) + '_in',
                                                shape=(indim, hidden_units[0]),
                                                initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                trainable=True,
                                                dtype=self.float_type)
                Bin = tf.compat.v1.get_variable(name=str(scope2B) + str(i_net) + '_in',
                                                shape=(hidden_units[0],),
                                                initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                dtype=self.float_type,
                                                trainable=True)
                self.Ws.append(Win)
                self.Bs.append(Bin)
                for i_layer in range(len(hidden_units) - 1):
                    stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                    W = tf.compat.v1.get_variable(name=str(scope2W) + str(i_net) + str(i_layer),
                                                  shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                                                  initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                  trainable=True,
                                                  dtype=self.float_type)
                    B = tf.compat.v1.get_variable(name=str(scope2B) + str(i_net) + str(i_layer),
                                                  shape=(hidden_units[i_layer + 1],),
                                                  initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                  trainable=True, dtype=self.float_type)
                    self.Ws.append(W)
                    self.Bs.append(B)

                # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
                stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
                Wout = tf.compat.v1.get_variable(name=str(scope2W) + str(i_net) + '_out',
                                                 shape=(hidden_units[-1], outdim),
                                                 initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                 trainable=True,
                                                 dtype=self.float_type)
                Bout = tf.compat.v1.get_variable(name=str(scope2B) + str(i_net) + '_out',
                                                 shape=(outdim,),
                                                 initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                 trainable=True,
                                                 dtype=self.float_type)

                self.Ws.append(Wout)
                self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=1.0):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """

        assert (len(scale) != 0)
        out_result = 0.0
        len2hidden = len(self.hidden_units)
        for i_net in range(self.num2subnets):
            i_in_layer = i_net*(len2hidden + 1)
            # ------ dealing with the input data ---------------
            H = tf.add(tf.matmul(inputs * scale[i_net], self.Ws[i_in_layer]), self.Bs[i_in_layer])
            H = self.actFunc_in(H)

            #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
            hidden_record = self.hidden_units[0]
            for i_layer in range(len2hidden - 1):
                i_hidden_layer = i_net * (len2hidden + 1) + i_layer + 1
                # W_hidden = self.Ws[i_hidden_layer]
                H_pre = H
                H = tf.add(tf.matmul(H, self.Ws[i_hidden_layer]), self.Bs[i_hidden_layer])
                H = self.actFunc(H)
                if self.hidden_units[i_layer + 1] == hidden_record:
                    H = H + H_pre
                hidden_record = self.hidden_units[i_layer + 1]

            i_out_layer = i_net * (len2hidden + 1) + len2hidden
            H = tf.add(tf.matmul(H, self.Ws[i_out_layer]), self.Bs[i_out_layer])
            out_result = out_result + self.actFunc_out(H)*(1.0/self.num2subnets)
        return out_result


class Scale_SubNets3D(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        num2subnets: the number of sub-nets
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5, num2subnets=5):
        super(Scale_SubNets3D, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName2in = actName2in
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.num2subnets = num2subnets

        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        self.Ws = []
        self.Bs = []
        with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
            stddev_WB = (2.0 / (indim + hidden_units[0])) ** varcoe
            Win = tf.compat.v1.get_variable(name=str(scope2W) + '_in',
                                            shape=(num2subnets, indim, hidden_units[0]),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                            trainable=True, dtype=self.float_type)
            Bin = tf.compat.v1.get_variable(name=str(scope2B) + '_in',
                                            shape=(num2subnets, 1, hidden_units[0]),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                            trainable=True, dtype=self.float_type)
            self.Ws.append(Win)
            self.Bs.append(Bin)
            for i_layer in range(len(hidden_units) - 1):
                stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                W = tf.compat.v1.get_variable(name=str(scope2W) + str(i_layer),
                                              shape=(num2subnets, hidden_units[i_layer], hidden_units[i_layer + 1]),
                                              initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                              trainable=True, dtype=self.float_type)
                B = tf.compat.v1.get_variable(name=str(scope2B) + str(i_layer),
                                              shape=(num2subnets, 1, hidden_units[i_layer + 1]),
                                              # shape=(num2subnets, hidden_units[i_layer + 1], 1),
                                              initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                              trainable=True, dtype=self.float_type)
                self.Ws.append(W)
                self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.compat.v1.get_variable(name=str(scope2W) + '_out',
                                             shape=(num2subnets, hidden_units[-1], outdim),
                                             initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                             trainable=True, dtype=self.float_type)
            Bout = tf.compat.v1.get_variable(name=str(scope2B) + '_out',
                                             shape=(num2subnets, 1, outdim),
                                             initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                             trainable=True, dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

            # stddev_WB = (2.0 / (num2subnets + 1)) ** varcoe
            # self.weight2subnets = tf.compat.v1.get_variable(name=str(scope2W) + '_subnet',
            #                                                 shape=(num2subnets, 1, 1),
            #                                                 initializer=tf.random_normal_initializer(stddev=stddev_WB),
            #                                                 trainable=True, dtype=self.float_type)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=1.0):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        assert (len(scale) != 0)
        expand_scale = np.expand_dims(scale, axis=-1)
        exapnd2_scale = np.expand_dims(expand_scale, axis=-1)
        exapnd2_scale = exapnd2_scale.astype(dtype=np.float32)
        len2hidden = len(self.hidden_units)

        H_in = tf.matmul(inputs, self.Ws[0])
        H_scale = tf.multiply(exapnd2_scale, H_in)
        H = tf.add(H_scale, self.Bs[0])
        H = self.actFunc_in(H)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len2hidden - 1):
            H_pre = H
            H = tf.add(tf.matmul(H, self.Ws[i_layer + 1]), self.Bs[i_layer + 1])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record and i_layer != 0:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        # out_result = tf.multiply(out_result, self.weight2subnets)
        out_result = tf.multiply(out_result, 1.0/exapnd2_scale)
        out_result = tf.reduce_mean(out_result, axis=0)
        return out_result


class Fourier_SubNets(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5, num2subnets=0.5):
        super(Fourier_SubNets, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName2in = actName2in
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.num2subnets = num2subnets

        self.type2float = type2float
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        self.Ws = []
        self.Bs = []
        for i_net in range(num2subnets):
            with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
                stddev_WB = (2.0 / (indim + hidden_units[0])) ** varcoe
                Win = tf.compat.v1.get_variable(name=str(scope2W) + str(i_net) + '_in',
                                                shape=(indim, hidden_units[0]),
                                                initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                trainable=True,
                                                dtype=self.float_type)
                Bin = tf.compat.v1.get_variable(name=str(scope2B) + str(i_net) + '_in',
                                                shape=(hidden_units[0],),
                                                initializer=tf.zeros_initializer(),
                                                dtype=self.float_type,
                                                trainable=False)
                self.Ws.append(Win)
                self.Bs.append(Bin)
                for i_layer in range(len(hidden_units)-1):
                    stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                    if i_layer == 0:
                        W = tf.compat.v1.get_variable(name=str(scope2W) + str(i_net) + str(i_layer),
                                                      shape=(hidden_units[i_layer] * 2, hidden_units[i_layer + 1]),
                                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                      trainable=True,
                                                      dtype=self.float_type)
                        B = tf.compat.v1.get_variable(name=str(scope2B) + str(i_net) + str(i_layer),
                                                      shape=(hidden_units[i_layer + 1],),
                                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                      trainable=True,
                                                      dtype=self.float_type)
                    else:
                        W = tf.compat.v1.get_variable(name=str(scope2W) + str(i_net) + str(i_layer),
                                                      shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                      trainable=True,
                                                      dtype=self.float_type)
                        B = tf.compat.v1.get_variable(name=str(scope2B) + str(i_net) + str(i_layer),
                                                      shape=(hidden_units[i_layer + 1],),
                                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                      trainable=True,
                                                      dtype=self.float_type)
                    self.Ws.append(W)
                    self.Bs.append(B)

                # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
                stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
                Wout = tf.compat.v1.get_variable(name=str(scope2W) + str(i_net) + '_out',
                                                 shape=(hidden_units[-1], outdim),
                                                 initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                 trainable=True,
                                                 dtype=self.float_type)
                Bout = tf.compat.v1.get_variable(name=str(scope2B) + str(i_net) + '_out',
                                                 shape=(outdim,),
                                                 initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                 trainable=True,
                                                 dtype=self.float_type)

                self.Ws.append(Wout)
                self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=1.0):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        assert (len(scale) != 0)
        out_result = 0.0
        len2hidden = len(self.hidden_units)
        for i_net in range(self.num2subnets):
            # ------ dealing with the input data ---------------
            i_in_layer = i_net * (len2hidden + 1)
            # H = tf.add(tf.matmul(inputs * scale[i_net], self.Ws[i_in_layer]), self.Bs[i_in_layer])
            H = tf.matmul(inputs * scale[i_net], self.Ws[i_in_layer])

            H = sFourier*tf.concat([tf.cos(H), tf.sin(H)], axis=-1)

            #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
            hidden_record = self.hidden_units[0]
            for i_layer in range(len2hidden - 1):
                i_hidden_layer = i_net * (len2hidden + 1) + i_layer + 1
                H_pre = H
                H = tf.add(tf.matmul(H, self.Ws[i_hidden_layer]), self.Bs[i_hidden_layer])
                H = self.actFunc(H)
                if self.hidden_units[i_layer + 1] == hidden_record and i_layer != 0:
                    H = H + H_pre
                hidden_record = self.hidden_units[i_layer + 1]

            i_out_layer = i_net * (len2hidden + 1) + len2hidden
            H = tf.add(tf.matmul(H, self.Ws[i_out_layer]), self.Bs[i_out_layer])
            out_result = out_result + self.actFunc_out(H)*(1.0/self.num2subnets)
        return out_result


class Fourier_SubNets3D(object):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32',
                 varcoe=0.5, num2subnets=0.5):
        super(Fourier_SubNets3D, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName2in = actName2in
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.num2subnets = num2subnets

        self.type2float = type2float
        if type2float == 'float32':
            self.float_type = tf.float32
        elif type2float == 'float64':
            self.float_type = tf.float64
        else:
            self.float_type = tf.float16

        self.Ws = []
        self.Bs = []
        with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
            stddev_WB = (2.0 / (indim + hidden_units[0])) ** varcoe
            Win = tf.compat.v1.get_variable(name=str(scope2W) + '_in',
                                            shape=(num2subnets, indim, hidden_units[0]),
                                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                            trainable=True, dtype=self.float_type)
            Bin = tf.compat.v1.get_variable(name=str(scope2B) + '_in',
                                            shape=(num2subnets, 1, hidden_units[0]),
                                            initializer=tf.zeros_initializer(),
                                            dtype=self.float_type, trainable=False)
            self.Ws.append(Win)
            self.Bs.append(Bin)
            for i_layer in range(len(hidden_units)-1):
                stddev_WB = (2.0 / (hidden_units[i_layer] + hidden_units[i_layer + 1])) ** varcoe
                if i_layer == 0:
                    W = tf.compat.v1.get_variable(name=str(scope2W) + str(i_layer),
                                                  shape=(num2subnets, hidden_units[i_layer]*2, hidden_units[i_layer+1]),
                                                  initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                  trainable=True, dtype=self.float_type)
                    B = tf.compat.v1.get_variable(name=str(scope2B) + str(i_layer),
                                                  shape=(num2subnets, 1, hidden_units[i_layer + 1]),
                                                  initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                  trainable=True, dtype=self.float_type)
                else:
                    W = tf.compat.v1.get_variable(name=str(scope2W) + str(i_layer),
                                                  shape=(num2subnets, hidden_units[i_layer], hidden_units[i_layer + 1]),
                                                  initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                  trainable=True, dtype=self.float_type)
                    B = tf.compat.v1.get_variable(name=str(scope2B) + str(i_layer),
                                                  shape=(num2subnets, 1, hidden_units[i_layer + 1]),
                                                  initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                                  trainable=True, dtype=self.float_type)
                self.Ws.append(W)
                self.Bs.append(B)

            # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
            stddev_WB = (2.0 / (hidden_units[-1] + outdim)) ** varcoe
            Wout = tf.compat.v1.get_variable(name=str(scope2W) + '_out',
                                             shape=(num2subnets, hidden_units[-1], outdim),
                                             initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                             trainable=True, dtype=self.float_type)
            Bout = tf.compat.v1.get_variable(name=str(scope2B) + '_out',
                                             shape=(num2subnets, 1, outdim),
                                             initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                             trainable=True, dtype=self.float_type)

            self.Ws.append(Wout)
            self.Bs.append(Bout)

            # stddev_WB = (2.0 / (num2subnets + 1)) ** varcoe
            # self.weight2subnets = tf.compat.v1.get_variable(name=str(scope2W) + '_subnet',
            #                                                 shape=(num2subnets, 1, 1),
            #                                                 initializer=tf.random_normal_initializer(stddev=stddev_WB),
            #                                                 trainable=True, dtype=self.float_type)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def __call__(self, inputs, scale=None, sFourier=1.0):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        assert (len(scale) != 0)
        expand_scale = np.expand_dims(scale, axis=-1)
        exapnd2_scale = np.expand_dims(expand_scale, axis=-1)
        exapnd2_scale = exapnd2_scale.astype(dtype=np.float32)
        len2hidden = len(self.hidden_units)
        # ------ dealing with the input data ---------------
        H_in = tf.matmul(inputs, self.Ws[0])

        H = tf.multiply(exapnd2_scale, H_in)

        H = sFourier*tf.concat([tf.cos(H), tf.sin(H)], axis=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len2hidden - 1):
            H_pre = H
            H = tf.add(tf.matmul(H, self.Ws[i_layer + 1]), self.Bs[i_layer + 1])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record and i_layer != 0:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        # out_result = tf.multiply(out_result, self.weight2subnets)
        # normalize_ceof = tf.nn.softmax(tf.exp(-1.0*exapnd2_scale))
        # out_result = tf.multiply(out_result, normalize_ceof)
        out_result = tf.multiply(out_result, 1.0/exapnd2_scale)
        out_result = tf.reduce_sum(out_result, axis=0)
        return out_result


def func2test_subNets():
    input_dim = 2
    out_dim = 1
    freq = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    hidden_layer = (4, 10, 5, 5, 4)
    # name2base_model = 'Fourier_subDNN'
    name2base_model = 'Scale_subDNN'
    actFun = 'tanh'
    if str.upper(name2base_model) == 'FOURIER_SUBDNN':
        model = Fourier_SubNets3D(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=name2base_model,
                                  actName2in=actFun, actName=actFun, num2subnets=8)
    elif str.upper(name2base_model) == 'SCALE_SUBDNN':
        model = Scale_SubNets3D(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=name2base_model,
                                actName2in=actFun, actName=actFun, num2subnets=8)
    batch_size = 10
    x = np.random.rand(batch_size, input_dim)

    with tf.device('/gpu:%s' % ('0')):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            X = tf.placeholder(tf.float32, name='XYit2train', shape=[None, input_dim])  # [N, D]
            Y = model(X, scale=freq)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        for i_epoch in range(5):
            sess.run(tf.global_variables_initializer())
            y = sess.run(Y, feed_dict={X: x})
            print('Y:', y)


def func2test_PureDNN():
    input_dim = 2
    out_dim = 1
    hidden_layer = (5, 10, 10, 15, 20)
    name2base_model = 'DNN'
    actFun = 'tanh'

    model = Dense_Net(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=name2base_model,
                      actName=actFun)
    batch_size = 10
    x = np.random.rand(batch_size, input_dim)
    freq = [1, 2, 3, 4, 5, 6, 7, 8]
    with tf.device('/gpu:%s' % ('0')):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            X = tf.placeholder(tf.float32, name='XYit2train', shape=[None, input_dim])  # [N, D]
            Y = model(X, scale=freq)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        for i_epoch in range(5):
            sess.run(tf.global_variables_initializer())
            y = sess.run(Y, feed_dict={X: x})
            print('Y:', y)


def func_test(x, in_dim=2, equa='eq1'):
    if in_dim == 1 and equa == 'eq1':
        out = np.sin(np.pi * x[:, 0]) + 0.1 * np.sin(3 * np.pi * x[:, 0]) + 0.01 * np.sin(10 * np.pi * x[:, 0])
    if in_dim == 1 and equa == 'eq2':
        out = np.sin(np.pi * x[:, 0]) + 0.1 * np.sin(3 * np.pi * x[:, 0]) + \
              0.05 * np.sin(10 * np.pi * x[:, 0]) + 0.01 * np.sin(50 * np.pi * x[:, 0])
    elif in_dim == 2:
        out = np.sin(x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]) + 0.1 * np.sin(5 * np.pi * x[:, 0] * x[:, 0] + 3 * np.pi * x[:, 1] * x[:, 1]) + \
              0.01 * np.sin(15 * np.pi * x[:, 0] * x[:, 0] + 20 * np.pi * x[:, 1] * x[:, 1])

    out = np.reshape(out, newshape=(-1, 1))
    return out


# This is an example for using the above module
class DNN2TEST(object):
    def __init__(self, input_dim=2, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='tanh',
                 name2actHidden='tanh', name2actOut='linear', factor2freq=None, sFourier=1.0, opt2regular_WB='L0',
                 type2numeric='float32', repeat_Highfreq=True):
        super(DNN2TEST, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_layer = hidden_layer
        self.Model_name = Model_name
        self.name2actIn = name2actIn
        self.name2actHidden = name2actHidden
        self.name2actOut = name2actOut
        self.factor2freq = factor2freq
        self.sFourier = sFourier
        self.opt2regular_WB = opt2regular_WB

        if str.upper(Model_name) == 'SCALE_DNN':
            self.DNN = Dense_ScaleNet(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer,
                                      name2Model=Model_name, actName2in=name2actIn, actName=name2actHidden,
                                      actName2out=name2actOut, type2float=type2numeric,
                                      repeat_high_freq=repeat_Highfreq)
        elif str.upper(Model_name) == 'SCALE_SUBDNN':
            self.DNN = Scale_SubNets3D(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer,
                                       name2Model=Model_name, actName2in=name2actIn, actName=name2actHidden,
                                       actName2out=name2actOut, type2float=type2numeric,
                                       repeat_high_freq=repeat_Highfreq, num2subnets=len(factor2freq))
        elif str.upper(Model_name) == 'FOURIER_DNN':
            self.DNN = Dense_FourierNet(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                                        actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut,
                                        type2float=type2numeric, repeat_high_freq=repeat_Highfreq)
        elif str.upper(Model_name) == 'FOURIER_FEATUREDNN':
            self.DNN = Fourier_FeatureDNN(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                                          actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut,
                                          type2float=type2numeric, repeat_high_freq=repeat_Highfreq, sigma=30.0)
        elif str.upper(Model_name) == 'MULTI_4FF_DNN':
            self.DNN = Multi_4FF_DNN(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                                     actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut,
                                     type2float=type2numeric, repeat_high_freq=repeat_Highfreq, sigma1=1.0,
                                     sigma2=10.0, sigma3=20.0, sigma4=50.0)
        elif str.upper(Model_name) == 'FOURIER_SUBDNN':
            self.DNN = Fourier_SubNets3D(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                                         actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut,
                                         type2float=type2numeric, repeat_high_freq=repeat_Highfreq,
                                         num2subnets=len(factor2freq))

        if type2numeric == 'float32':
            self.float_type = tf.float32
        elif type2numeric == 'float64':
            self.float_type = tf.float64
        elif type2numeric == 'float16':
            self.float_type = tf.float16

    def forward(self, x_input):
        out = self.DNN(x_input, scale=self.factor2freq)
        return out

    def get_sum2wB(self):
        sum2WB = self.DNN.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        return sum2WB

    def cal_l2loss(self, x_input=None, y_input=None):
        out = self.DNN(x_input, scale=self.factor2freq)
        squre_loss = tf.multiply(y_input - out, y_input - out)
        loss = tf.reduce_mean(squre_loss, axis=0)
        return loss, out


def func2test_DNN():
    batch_size = 100
    # dim_in = 1
    dim_in = 2
    dim_out = 1
    act_func2In = 'tanh'
    # act_func2In = 'sin'
    # act_func2In = 'Enhance_tanh'

    act_func2Hidden = 'tanh'
    # act_func2Hidden = 'Enhance_tanh'
    # act_func2Hidden = 'sin'
    # act_func2Hidden = 'sinAddcos'
    # act_func2Hidden = 'gelu'

    act_func2Out = 'linear'

    # hidden_list = (10, 15, 10, 10, 5)  # 10+150+150+100+50+5= 465
    hidden_list = (15, 20, 10, 10, 5)  # 15+300+200+100+50+5= 670
    # hidden_list = (20, 15, 10, 10, 5)  # 20+300+150+100+50+5= 625
    freq = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], dtype=np.float32)

    # model_name = 'Fourier_DNN'
    # model_name = 'Fourier_SubDNN'
    # model_name = 'Scale_SubDNN'
    # model_name = 'Fourier_FeatureDNN'
    # model_name = 'Scale_DNN'
    model_name = 'Multi_4FF_DNN'

    if model_name == 'Fourier_DNN':
        # hidden_list = (50, 80, 60, 60, 40)   # 50 + 4000 + 4800 + 3600 + 2400 + 40 = 14890
        hidden_list = (60, 80, 60, 60, 40)     # 60 + 4800 + 4800 + 3600 + 2400 + 40 = 15700
    elif model_name == 'Fourier_FeatureDNN':
        # hidden_list = (50, 80, 60, 60, 40)   # 50 + 4000 + 4800 + 3600 + 2400 + 40 = 14890
        hidden_list = (60, 80, 60, 60, 40)     # 60 + 4800 + 4800 + 3600 + 2400 + 40 = 15700
    elif model_name == 'Multi_4FF_DNN':
        # hidden_list = (50, 80, 60, 60, 40)   # 50 + 4000 + 4800 + 3600 + 2400 + 40 = 14890
        hidden_list = (60, 80, 60, 60, 20)     # 60 + 4800 + 4800 + 3600 + 2400 + 40 = 15700
    elif model_name == 'Scale_DNN':
        # hidden_list = (100, 80, 60, 60, 40)
        hidden_list = (120, 80, 60, 60, 40)

    init_lr = 0.01
    lr_decay = 0.0001
    max_it = 100000
    highFreq_repeat = True

    model = DNN2TEST(input_dim=dim_in, out_dim=dim_out, hidden_layer=hidden_list, Model_name=model_name,
                     name2actIn=act_func2In, name2actHidden=act_func2Hidden, name2actOut=act_func2Out, factor2freq=freq,
                     sFourier=1.0, repeat_Highfreq=highFreq_repeat)

    global_steps = tf.compat.v1.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % 0):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            X = tf.placeholder(tf.float32, name='X2train', shape=[None, dim_in])  # [N, D]
            Y = tf.placeholder(tf.float32, name='Y2train', shape=[None, dim_out])  # [N, D]
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            loss2data, prediction = model.cal_l2loss(x_input=X, y_input=Y)
            sum2wb = model.get_sum2wB()
            loss = loss2data + sum2wb

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)

    arr2epoch = []
    arr2loss = []
    temp_lr = init_lr
    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i_epoch in range(max_it):
            temp_lr = temp_lr * (1 - lr_decay)
            x = np.random.rand(batch_size, dim_in)
            x = x.astype(dtype=np.float32)
            y = func_test(x, in_dim=dim_in, equa='eq2')
            _, loss_it = sess.run([train_my_loss, loss], feed_dict={X: x, Y:y, in_learning_rate:temp_lr})

            if i_epoch % 100 == 0:
                print('i_epoch ------- loss:', i_epoch, loss_it)
                print("第%d个epoch的学习率：%.10f" % (i_epoch, temp_lr))
                arr2epoch.append(int(i_epoch/100))
                arr2loss.append(loss_it)

    if model_name == 'DNN':
        plt.figure()
        ax = plt.gca()
        plt.plot(arr2loss, 'b-.', label='loss')
        plt.xlabel('epoch/100', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.legend(fontsize=18)
        plt.title('DNN')
        ax.set_yscale('log')
        plt.show()
    else:
        plt.figure()
        ax = plt.gca()
        plt.plot(arr2loss, 'b-.', label='loss')
        plt.xlabel('epoch/100', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.legend(fontsize=18)
        plt.title('Fourier_DNN')
        ax.set_yscale('log')
        plt.show()


if __name__ == "__main__":
    # func2test_subNets()

    func2test_DNN()
# coding = utf-8

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet.conv as conv
import theano.tensor.signal.pool as pool

class Hidden_layer:
    def __init__(self, input, n_in, n_out, rng):
        self.rng = rng
        self.input = input
        w_value = rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), \
                              high=np.sqrt(6. / (n_in + n_out)), \
                              size=(n_in, n_out))
        self.W = theano.shared(value=np.array(w_value, dtype=theano.config.floatX), \
                               name='h_w',
                               borrow=True)
        self.b = theano.shared(value=np.zeros(n_out, dtype=theano.config.floatX), \
                               name='h_b',
                               borrow=True)
        self.params = [self.W, self.b]
        self.outputs = T.tanh(T.dot(self.input, self.W) + self.b)


class Logistic_layer:
    def __init__(self, input, n_in, n_out, rng):
        self.rng = rng
        self.input = input
        w_value = rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), \
                              high=np.sqrt(6. / (n_in + n_out)), \
                              size=(n_in, n_out))
        self.W = theano.shared(value=np.array(w_value, dtype=theano.config.floatX), \
                               name='L_w',
                               borrow=True)
        self.b = theano.shared(value=np.zeros(n_out, dtype=theano.config.floatX), \
                               name='L_b',
                               borrow=True)
        self.params = [self.W, self.b]
        self.outputs = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
        self.pred_y = T.argmax(self.outputs, axis=1)

    def cost(self, y):
        return -T.mean(T.log(self.outputs)[T.arange(y.shape[0]), y])

    def accurcy(self, y):
        acc = T.mean(T.neq(self.pred_y, y))
        return acc


class conv_pool_layer:
    def __init__(self, input, rng, input_shape, filter_shape, pool_size=(2, 2)):
        self.input = input
        self.rng = rng
        fan_in = np.prod(filter_shape[1:])
        w_value = np.array(self.rng.uniform(low=-3.0 / fan_in, high=3.0 / fan_in, size=filter_shape),
                           dtype=theano.config.floatX)
        self.W = theano.shared(value=w_value, name='c_w', borrow=True)
        b_value = np.zeros(filter_shape[0], dtype=theano.config.floatX)
        self.b = theano.shared(value=b_value, name='c_b', borrow=True)
        self.params = [self.W, self.b]
        conv_out = conv.conv2d(self.input, self.W, image_shape=input_shape,
                               filter_shape=filter_shape)
        max_pool_out = pool.pool_2d(conv_out, pool_size)
        self.outputs = T.tanh(max_pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))


class conv_3x3:
    def __init__(self, input, rng, input_shape, filter_shape_1, filter_shape_2, pool_size=(2, 2)):
        self.input = input
        self.rng = rng
        fan_in_1 = np.prod(filter_shape_1[1:])
        fan_in_2 = np.prod(filter_shape_2[1:])
        filter_shape_3 = (filter_shape_2[0], filter_shape_2[0], filter_shape_2[2], filter_shape_2[2])
        w_value = np.array(self.rng.uniform(low=-3.0 / fan_in_1, high=3.0 / fan_in_1, size=filter_shape_1),
                           dtype=theano.config.floatX)
        w_2value = np.array(self.rng.uniform(low=-3.0 / fan_in_2, high=3.0 / fan_in_2, size=filter_shape_2),
                            dtype=theano.config.floatX)
        w_3value = np.array(self.rng.uniform(low=-3.0 / fan_in_2, high=3.0 / fan_in_2, size=filter_shape_3),
                            dtype=theano.config.floatX)
        self.W_1 = theano.shared(value=w_value, name='1_w', borrow=True)
        self.W_2 = theano.shared(value=w_2value, name='2_w', borrow=True)
        self.W_3 = theano.shared(value=w_3value, name='3_w', borrow=True)
        b1_value = np.zeros(filter_shape_1[0], dtype=theano.config.floatX)
        b2_value = np.zeros(filter_shape_2[0], dtype=theano.config.floatX)
        self.b1 = theano.shared(value=b1_value, name='1_b', borrow=True)
        self.b2 = theano.shared(value=b2_value, name='2_b', borrow=True)
        self.params = [self.W_1, self.b2, self.b1, self.W_3, self.W_2]
        conv_out_1 = conv.conv2d(self.input, self.W_1, image_shape=input_shape,
                                 filter_shape=filter_shape_1)
        output_1 = T.tanh(conv_out_1 + self.b1.dimshuffle('x', 0, 'x', 'x'))
        # the input of conv_out_2 should
        conv_out_2 = conv.conv2d(output_1, self.W_2, filter_shape=filter_shape_2)
        # max_pool_out = downsample.max_pool_2d(conv_out_2, pool_size)
        max_pool_out = conv.conv2d(conv_out_2, self.W_3, filter_shape=filter_shape_3, subsample=(2, 2))
        self.outputs = T.tanh(max_pool_out + self.b2.dimshuffle('x', 0, 'x', 'x'))


class global_average_pool:
    def __init__(self, input, pool_size):
        self.input = input
        self.outputs = pool.pool_2d(self.input, pool_size, mode='average_exc_pad')


class network_in_network:
    def __init__(self, inputs, rng, filter_1, mlp_1, mlp_2, mlp_3):
        w1_value = rng.normal(size=filter_1)
        w2_value = rng.normal(size=mlp_1)
        w3_value = rng.normal(size=mlp_2)
        w4_value = rng.normal(size=mlp_3)
        b1 = np.zeros(mlp_1[0], dtype=theano.config.floatX)
        b2 = np.zeros(mlp_2[0], dtype=theano.config.floatX)
        b3 = np.zeros(mlp_3[0], dtype=theano.config.floatX)
        self.inputs = inputs
        self.w1 = theano.shared(value=np.array(w1_value, dtype=theano.config.floatX), name='w1', borrow=True)
        self.w2 = theano.shared(value=np.array(w2_value, dtype=theano.config.floatX), name='w2', borrow=True)
        self.w3 = theano.shared(value=np.array(w3_value, dtype=theano.config.floatX), name='w3', borrow=True)
        self.w4 = theano.shared(value=np.array(w4_value, dtype=theano.config.floatX), name='w4', borrow=True)
        self.b1 = theano.shared(value=b1, name='b1', borrow=True)
        self.b2 = theano.shared(value=b2, name='b2', borrow=True)
        self.b3 = theano.shared(value=b3, name='b3', borrow=True)
        conv_out = conv.conv2d(self.inputs, self.w1)
        mlp_1 = T.nnet.relu(conv.conv2d(conv_out, self.w2) + self.b1.dimshuffle('x', 0, 'x', 'x'))
        mlp_2 = T.nnet.relu(conv.conv2d(mlp_1, self.w3) + self.b2.dimshuffle('x', 0, 'x', 'x'))
        mlp_3 = T.nnet.relu(conv.conv2d(mlp_2, self.w4) + self.b3.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.w1, self.w2, self.w3, self.w4, self.b1, self.b2, self.b3]
        self.outputs = mlp_3

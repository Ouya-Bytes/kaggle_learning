# coding = utf-8

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet.conv as conv
import theano.tensor.signal.downsample as downsample


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
        max_pool_out = downsample.max_pool_2d(conv_out, pool_size)
        self.outputs = T.tanh(max_pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))


class conv_3x3:
    def __init__(self, input, rng, input_shape, filter_shape_1, filter_shape_2, pool_size=(2, 2)):
        self.input = input
        self.rng = rng
        fan_in_1 = np.prod(filter_shape_1[1:])
        fan_in_2 = np.prod(filter_shape_2[1:])
        w_value = np.array(self.rng.uniform(low=-3.0 / fan_in_1, high=3.0 / fan_in_1, size=filter_shape_1),
                           dtype=theano.config.floatX)
        w_2value = np.array(self.rng.uniform(low=-3.0 / fan_in_2, high=3.0 / fan_in_2, size=filter_shape_2),
                            dtype=theano.config.floatX)
        self.W_1 = theano.shared(value=w_value, name='c_w', borrow=True)
        self.W_2 = theano.shared(value=w_2value, name='c_w', borrow=True)
        b1_value = np.zeros(filter_shape_1[0], dtype=theano.config.floatX)
        b2_value = np.zeros(filter_shape_2[0], dtype=theano.config.floatX)
        self.b1 = theano.shared(value=b1_value, name='c_b', borrow=True)
        self.b2 = theano.shared(value=b2_value, name='c_b', borrow=True)
        self.params = [self.W_1, self.b2, self.W_2]
        conv_out_1 = conv.conv2d(self.input, self.W_1, image_shape=input_shape,
                                 filter_shape=filter_shape_1)
        output_1 = T.tanh(conv_out_1 + self.b1.dimshuffle('x', 0, 'x', 'x'))
        # the input of conv_out_2 should
        conv_out_2 = conv.conv2d(conv_out_1, self.W_2, filter_shape=filter_shape_2)
        max_pool_out = downsample.max_pool_2d(conv_out_2, pool_size)
        self.outputs = T.tanh(max_pool_out + self.b2.dimshuffle('x', 0, 'x', 'x'))

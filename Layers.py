# coding = utf-8

import theano
import theano.tensor as T
import numpy as np
import theano.tensor.nnet.conv as conv
import theano.tensor.signal.downsample as downsample


class Hidden_layer:
    def __init__(self, input, n_in, n_out, rng):
        self.rng = rng
        self.input = input
        w_value = rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), \
                              high=np.sqrt(6. / (n_in + n_out)), \
                              size=(n_in, n_out))
        self.W = theano.shared(value=np.array(w_value,  dtype=theano.config.floatX), \
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
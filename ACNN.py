# coding = utf-8

import theano.tensor as T

import Layers


class Al_cnn:
    def __init__(self, input, batch_size, rng, n_kernels):
        self.input = input
        self.conv_pool_1 = Layers.conv_3x3(
            input=self.input,
            rng=rng,
            input_shape=(batch_size, 1, 28, 28),
            filter_shape_1=(n_kernels[0], 1, 3, 3),
            filter_shape_2=(n_kernels[0], n_kernels[0], 3, 3),
            pool_size=(2, 2)
        )
        self.conv_pool_2 = Layers.conv_3x3(
            input=self.conv_pool_1.outputs,
            rng=rng,
            input_shape=(batch_size, n_kernels[0], 11, 11),
            filter_shape_1=(n_kernels[1], n_kernels[0], 3, 3),
            filter_shape_2=(n_kernels[1], n_kernels[1], 3, 3),
            pool_size=(2, 2)
        )
        hidden_layer_input = self.conv_pool_2.outputs.flatten(2)
        self.hidden_layer_1 = Layers.Hidden_layer(
            input=hidden_layer_input,
            rng=rng,
            n_in=n_kernels[1] * 3 * 3,
            n_out=192
        )
        self.softmax_layer = Layers.Logistic_layer(
            input=self.hidden_layer_1.outputs,
            rng=rng,
            n_in=192,
            n_out=10
        )
        self.results = self.softmax_layer.pred_y
        self.params = self.conv_pool_1.params + self.conv_pool_2.params + self.hidden_layer_1.params + self.softmax_layer.params
    def cost_updates(self, y, learning_rate):
        acc = self.softmax_layer.accurcy(y)
        cost = self.softmax_layer.cost(y)
        grad_params = T.grad(cost, self.params)
        updates = [(param, param - learning_rate*g_param) for param, g_param in zip(self.params, grad_params)]
        return [cost, acc, updates]


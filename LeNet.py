# coding = utf-8

import theano
import theano.tensor as T
import Layers
class LeNet:
    def __init__(self, input, batch_size, rng, n_kernels):
        self.input = input
        self.conv_pool_1 = Layers.conv_pool_layer(
            input=self.input,
            rng=rng,
            input_shape=(batch_size, 1, 28, 28),
            filter_shape=(n_kernels[0], 1, 5, 5),
            pool_size=(2, 2)
        )
        self.conv_pool_2 = Layers.conv_pool_layer(
            input=self.conv_pool_1.outputs,
            rng=rng,
            input_shape=(batch_size, n_kernels[0], 12, 12),
            filter_shape=(n_kernels[1], n_kernels[0], 3, 3),
            pool_size=(2, 2)
        )
        hidden_layer_input = self.conv_pool_2.outputs.flatten(2)
        self.hidden_layer_1 = Layers.Hidden_layer(
            input=hidden_layer_input,
            rng=rng,
            n_in=n_kernels[1]*5*5,
            n_out=500
        )
        self.softmax_layer = Layers.Logistic_layer(
            input=self.hidden_layer_1.outputs,
            rng=rng,
            n_in=500,
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


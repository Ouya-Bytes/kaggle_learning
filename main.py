# coding = utf-8
import load_data, LeNet, generate_function
import theano.tensor as T
import theano, Layers
import numpy as np
import timeit

# load data
train_x, train_y, test_x, test_y = load_data.split_data()
train_x, train_y = load_data.shared_data(train_x, train_y)
test_x, test_y = load_data.shared_data(test_x, test_y)
X = T.matrix('input', dtype=theano.config.floatX)
y = T.ivector('labels')
index = T.lscalar('index')
batch_size = 20
learning_rate = 0.01
train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
test_batches = test_x.get_value(borrow=True).shape[0] // batch_size
rng = np.random.RandomState(1234)

# create model
layer0_input = X.reshape((batch_size, 1, 28, 28))
conv_net = LeNet.LeNet(
    input=layer0_input,
    batch_size=batch_size,
    rng=rng,
    n_kernels=[4, 6]
)
[cost, acc, updates] = conv_net.cost_updates(y, learning_rate=learning_rate)
# gererate model
givens_train={
        X: train_x[index * batch_size:(index + 1) * batch_size],
        y: train_y[index * batch_size:(index + 1) * batch_size]
    }
givens_test={
        X: test_x[index * batch_size:(index + 1) * batch_size],
        y: test_y[index * batch_size:(index + 1) * batch_size]
    }
givens = [givens_train, givens_test]
function = generate_function.function([index], cost, updates, givens)
[train_model, test_model]= function.model

def train_test(epoches):
    train_acc = []
    test_acc = []
    for epoch in xrange(epoches):
        t1 = timeit.default_timer()
        for batch_index in xrange(train_batches):
            train_acc.append(train_model(batch_index))
        print 'epoch={}, accuracy={}, cost={}, time={}'.format(epoch,
                                                               np.mean(np.sum(np.array(train_acc)[1])),
                                                               np.mean(np.sum(np.array(train_acc)[0])),
                                                               timeit.default_timer() - t1
                                                               )


if __name__ == '__main__':
    train_test(50)

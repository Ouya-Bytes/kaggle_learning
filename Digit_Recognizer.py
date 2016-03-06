# coding = utf-8
import cPickle
import csv
import timeit

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from sklearn import preprocessing

import LeNet
import generate_function
import load_data

min_max = preprocessing.MinMaxScaler()
# load data
train_x, train_y, test_x = load_data.train_test('./dataset/digit/train.csv', './dataset/digit/test.csv')
# normalizer value
train_x = min_max.fit_transform(train_x)
test_x = min_max.fit_transform(test_x)
train_x, train_y = load_data.shared_data(train_x, train_y)
X = T.matrix('input', dtype=theano.config.floatX)
y = T.ivector('labels')
index = T.lscalar('index')
batch_size = 20
learning_rate = 0.01
lamb = 0.001
train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
rng = np.random.RandomState(1234)

# create model
layer0_input = X.reshape((batch_size, 1, 28, 28))
conv_net = LeNet.LeNet(
    input=layer0_input,
    batch_size=batch_size,
    rng=rng,
    n_kernels=[4, 6]
)
[cost, acc, updates] = conv_net.cost_updates(y, learning_rate=learning_rate, lamb=lamb)
# gererate model
givens_train = {
    X: train_x[index * batch_size:(index + 1) * batch_size],
    y: train_y[index * batch_size:(index + 1) * batch_size]
}
givens = [givens_train]
function = generate_function.function([index], [cost, acc], conv_net.results, updates, givens)
[train_model] = function.model

def train_test(epoches):
    train_cost_value = []
    train_acc_value = []
    test_mean = 0
    best_acc = np.inf
    for epoch in xrange(epoches):
        train_acc = []
        t1 = timeit.default_timer()
        for batch_index in xrange(train_batches):
            train_acc.append(train_model(batch_index))
        print 'epoch={}, train_accuracy={}, valid_accuracy={}, cost={}, time={}'.format(epoch,
                                                                                        np.mean(
                                                                                            np.array(train_acc)[1]),
                                                                                        test_mean,
                                                                                        np.mean(
                                                                                            np.array(train_acc)[0]),
                                                                                        timeit.default_timer() - t1
                                                                                        )
        if np.mean(np.array(train_acc)[0]) < best_acc:
            best_acc = np.mean(np.array(train_acc)[0])
            with open('best_model_all.pkl', 'w') as f:
                cPickle.dump(conv_net, f)
                print 'save model...at time={}, epoch={}, best_cost={}'.format(timeit.default_timer(), epoch, best_acc)
        train_cost_value.append(np.mean(np.array(train_acc)[0]))
        train_acc_value.append(np.mean(np.array(train_acc)[1]))
    x = [i for i in xrange(epoches)]
    plt.figure()
    plt.scatter(x, train_acc_value)
    plt.show()
def predict():
    conv_net = cPickle.load(open('best_model_all.pkl', 'r'))
    pred_model = theano.function(
        inputs=[conv_net.input],
        outputs=conv_net.results
    )
    rows, cols = test_x.shape
    n_batches = rows / batch_size
    pred_value = [pred_model(test_x[i * batch_size:(i + 1) * batch_size].reshape((batch_size, 1, 28, 28))) for i in
                  xrange(n_batches)]
    result = np.concatenate(pred_value)
    print 'predict:{}'.format(len(result))
    with open('result_all.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageId', 'Label'])
        for i, item in enumerate(result):
            writer.writerow([i + 1, item])
    print 'complete...'
if __name__ == '__main__':
    train_test(250)
    predict()

# coding = utf-8
import load_data, LeNet, generate_function
import theano.tensor as T
import theano, Layers, cPickle
import numpy as np
import timeit, csv
import matplotlib.pyplot as plt

# load data
train_x, train_y, test_x = load_data.train_test('./dataset/digit/train.csv', './dataset/digit/test.csv')
train_x, train_y, valid_x, valid_y = load_data.split_data(train_x, train_y)
train_x, train_y = load_data.shared_data(train_x, train_y)
valid_x, valid_y = load_data.shared_data(valid_x, valid_y)
X = T.matrix('input', dtype=theano.config.floatX)
y = T.ivector('labels')
index = T.lscalar('index')
batch_size = 20
learning_rate = 0.01
train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size
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
givens_train = {
    X: train_x[index * batch_size:(index + 1) * batch_size],
    y: train_y[index * batch_size:(index + 1) * batch_size]
}
givens_valid = {
    X: valid_x[index * batch_size:(index + 1) * batch_size],
    y: valid_y[index * batch_size:(index + 1) * batch_size]
}
givens = [givens_train, givens_valid]
function = generate_function.function([index], [cost, acc], updates, givens)
[train_model, valid_model] = function.model


def train_test(epoches):
    train_cost_value = []
    train_acc_value = []
    test_mean = 0
    best_acc = np.inf
    test_frequency = 3
    for epoch in xrange(epoches):
        train_acc = []
        t1 = timeit.default_timer()
        for batch_index in xrange(train_batches):
            train_acc.append(train_model(batch_index))
            iter = (epoch - 1) * batch_size + batch_index
            '''
            if iter % test_frequency == 0:
                test_acc = [valid_model(i) for i in xrange(valid_batches)]
                test_mean = np.mean(np.array(test_acc)[1])
                '''
        print 'epoch={}, train_accuracy={}, valid_accuracy={}, cost={}, time={}'.format(epoch,
                                                                                        np.mean(
                                                                                            np.array(train_acc)[1]),
                                                                                        test_mean,
                                                                                        np.mean(
                                                                                            np.array(train_acc)[0]),
                                                                                        timeit.default_timer() - t1
                                                                                        )
        train_cost_value.append(np.mean(np.array(train_acc)[0]))
        train_acc_value.append(np.mean(np.array(train_acc)[1]))

    valid_acc = [valid_model(i) for i in xrange(valid_batches)]
    print 'test_accuracy = {}'.format(np.mean(np.array(valid_acc)[1]))
    x = [i for i in xrange(epoches)]
    plt.figure()
    plt.scatter(x, train_acc_value)
    plt.show()
    with open('best_model.pkl', 'w') as f:
        cPickle.dump(conv_net, f)
    print 'save model...at time={}'.format(timeit.default_timer())


def predict():
    conv_net = cPickle.load(open('best_model.pkl', 'r'))
    pred_model = theano.function(
        inputs=[conv_net.input],
        outputs=conv_net.results
    )
    rows, cols = test_x.shape
    n_batches = rows/batch_size
    pred_value = [pred_model(test_x[i*batch_size:(i+1)*batch_size].reshape((batch_size, 1, 28, 28))) for i in xrange(n_batches)]
    result = np.concatenate(pred_value)
    print 'predict:{}'.format(len(result))
    with open('result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageId', 'Label'])
        for i, item in enumerate(result):
            writer.writerow([i+1, item])
    print 'complete...'
if __name__ == '__main__':
    train_test(1)
    predict()

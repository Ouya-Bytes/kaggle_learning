# coding = utf-8
import csv, cPickle, os, theano
import theano.tensor as T
from sklearn.cross_validation import train_test_split
import numpy as np
def load_data(filename):
    f = open(filename)
    data = csv.reader(f)
    dataset = []
    count = 0
    for line in data:
        if count==0:
            count += 1
            continue
        else:
            temp = [int(item) for item in line]
            dataset += [temp]
    return dataset

def split_data(path = './dataset/digit/train.csv'):
    dataset = load_data(path)
    dataset = np.array(dataset, dtype=np.float32)
    X = dataset[:, 1:]
    y = dataset[:, 0]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)
    return (train_x, train_y, test_x, test_y)

def shared_data(X, y):
    X_shared = theano.shared(value=np.array(X, dtype=theano.config.floatX))
    y_shared = theano.shared(value=np.array(y, dtype=theano.config.floatX))
    return X_shared, T.cast(y_shared, 'int32')

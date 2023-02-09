#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q3.1a
        y_hat = self.predict(x_i)

        if y_hat != y_i:
            self.W[y_i]=self.W[y_i]+x_i
            self.W[y_hat]=self.W[y_hat]-x_i

        


class LogisticRegression(LinearModel):

    def softmax(self,z):
        expu = np.exp(z)
        return expu/np.sum(expu)

    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q3.1b
        rows, column = (self.W).shape
        y_one_hot = np.zeros(rows)
        y_one_hot[y_i] = 1
        q_i = self.softmax(self.W @ x_i)
        grad_i = np.outer(q_i - y_one_hot, x_i)
 
        self.W = self.W - learning_rate*grad_i


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, n_layers):
        # Initialize an MLP with a single hidden layer.
        self.W1 = np.random.normal(0.1,0.1,(hidden_size,n_features))
        self.W2 = np.random.normal(0.1,0.1,(n_classes,hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(n_classes)

    def relu(self, x):
        return np.maximum(x, 0)

    def softmax(self,z):
        z -= z.max()
        expu = np.exp(z)
        return expu/np.sum(expu)

    def d_relu(self,x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        predicted_labels = np.zeros(len(X))
        count = 0

        for x_i in X:

            x_i = np.asarray(x_i)
            h0 = x_i
            z1 = np.dot(self.W1, h0) + self.b1
            h1 = self.relu(z1)
            z2 = np.dot(self.W2,h1) + self.b2
            p = self.softmax(z2)
            predicted_labels[count] = p.argmax(axis=0)
            count = count + 1

        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        
        for x_i, y_i in zip(X, y):
            x_i = np.asarray(x_i)
            y_i = np.asarray(y_i)
            rows, column = (self.W2).shape
            y_one_hot = np.zeros(rows)
            y_one_hot[y_i] = 1
           
            #Foward propagation 
            h0 = x_i
            z1 = np.dot(self.W1, h0) + self.b1
            h1 = self.relu(z1)
            z2 = np.dot(self.W2,h1) + self.b2
            p = self.softmax(z2)

            # Backpropagation
            # Hidden layer
            g_out = p-y_one_hot
            g_weight_2 = np.dot(g_out[:,None], h1[None,:])
            g_bias_2 = g_out
            g_hidden_layer_2 = np.dot((self.W2).T, g_out)
            g_hidden_bef_2 = g_hidden_layer_2*self.d_relu(h1)

            # Input layer
            g_weight_1 = np.dot(g_hidden_bef_2[:,None], x_i[None,:])
            g_bias_1 = g_hidden_bef_2

            self.W1 = self.W1 - learning_rate*g_weight_1
            self.b1 = self.b1 - learning_rate*g_bias_1

            self.W2 = self.W2 - learning_rate*g_weight_2
            self.b2 = self.b2 - learning_rate*g_bias_2
    



def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    print('Final Test acc: %.4f' % (model.evaluate(test_X, test_y)))
    print('Final Validation acc: %.4f' % (model.evaluate(dev_X, dev_y)))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()

"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

        self.decay_rate = 0.01

        # to test out different hyper-params consistently
        np.random.seed(90210)

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        N, D = X_train.shape
        scores = X_train @ self.w.T  # (N, C)
        grads = self.reg_const * self.w

        for i in range(0, N):
            # should i be calculating scores on every iteration over the training set?
            # also consider batching.
            y_i = y_train[i]
            for c in range(0, self.n_class):
                y_i_score = scores[i][y_i]
                c_score = scores[i][c]
                if y_i != c and y_i_score < 1 + c_score:
                    grads[y_i] -= X_train[i]
                    grads[c] += X_train[i]

        return grads

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape

        self.w = np.random.uniform(-1, 1, (self.n_class, D)) * 0.01

        for epoch in range(self.epochs):
            grad = self.calc_gradient(X_train, y_train)
            current_lr = self.lr / (1 + self.decay_rate * epoch)
            self.w -= current_lr * grad

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        N, D = X_test.shape
        # W.shape = (C, D)
        scores = X_test @ self.w.T
        y_pred = scores.argmax(1)

        return y_pred

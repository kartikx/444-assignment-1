"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.decay_rate = 0.01

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        N, D = X_train.shape
        logits = X_train @ self.w.T  # N, C

        max_logits = np.max(logits, axis=1, keepdims=True)
        logits = logits - max_logits

        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # N, C

        grads = np.zeros_like(self.w)  # C, D

        for i in range(0, N):
            y_i = y_train[i]  # (1, )
            x_i = X_train[i]  # (d, )
            for c in range(0, self.n_class):
                score = probs[i][c]
                if c == y_i:
                    grads[c] += (score - 1) * x_i
                else:
                    grads[c] += score * x_i

        grads /= N

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

        return

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

        N, D = X_test.shape
        logits = X_test @ self.w.T  # N, C

        max_logits = np.max(logits, axis=1, keepdims=True)
        logits -= max_logits

        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # N, C

        return np.argmax(probs, axis=1)

"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # changed at bottom.
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.decay_rate = 0.05

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        # Hint: To prevent numerical overflow, try computing the sigmoid for positive numbers and negative numbers separately.
        #       - For negative numbers, try an alternative formulation of the sigmoid function.
        sig_z = np.zeros_like(z)

        for i in range(z.shape[0]):
            if (z[i] >= 0):
                sig_z[i] = 1 / (1 + np.exp(-z[i]))
            else:
                sig_z[i] = np.exp(z[i]) / (1 + np.exp(z[i]))

        return sig_z

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the logistic regression update rule as introduced in lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01.
        - This initialization prevents the weights from starting too large,
        which can cause saturation of the sigmoid function

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape

        self.w = np.random.uniform(-1, 1, (1, D)) * 0.01

        for epoch in range(self.epochs):
            # current_lr = self.lr / (1 + self.decay_rate * epoch)
            # for i in range(N):
            #     scores = self.sigmoid(X_train @ self.w.T)
            #     y_i = y_train[i]
            #     x_i = X_train[i]
            #     grad = (scores[i] - y_i) * x_i
            #     grad /= N
            #     self.w -= current_lr * grad
            # Recalculate scores with updated weights

            scores = self.sigmoid(X_train @ self.w.T).flatten()

            # Compute current learning rate once per epoch
            current_lr = self.lr / (1 + self.decay_rate * epoch)

            # Compute the gradient vectorized over all samples
            grad = ((scores - y_train) @ X_train) / N

            # Update weights
            self.w -= current_lr * grad

        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:exce
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        scores = self.sigmoid(X_test @ self.w.T).flatten()
        output = (scores > 0.5).astype(int)
        print(output.shape)
        return output

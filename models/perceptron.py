"""Perceptron model."""

import numpy as np

print_validation = 10


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.decay_rate = 0.01

        # to test out different hyper-params consistently
        np.random.seed(90210)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the perceptron update rule as introduced in the Lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels

        self.w = np.random.uniform(-1, 1,
                                   (self.n_class, X_train.shape[1])) * 0.01

        N = X_train.shape[0]

        for epoch in range(self.epochs):
            total_loss = 0.0
            for i in range(N):
                x_i = X_train[i]
                y_i = y_train[i]

                # todo think about doing without the for loop via broadcasting
                scores = np.dot(self.w, x_i)

                example_loss = 0.0

                for c in range(self.n_class):
                    example_loss += max(0, scores[c] - scores[y_i])
                    if scores[c] > scores[y_i]:
                        self.w[y_i] += self.lr * x_i
                        self.w[c] -= self.lr * x_i

                total_loss += example_loss

            if epoch % print_validation == 0:
                print("Loss: ", total_loss / N)
        """

        N, D = X_train.shape
        # Initialize weights and bias
        self.w = np.random.uniform(-1, 1, (self.n_class, D)) * 0.01
        self.b = np.zeros(self.n_class)
        self.delta = 0

        for epoch in range(self.epochs):
            total_loss = 0
            current_lr = self.lr / (1 + self.decay_rate * epoch)
            for i in range(N):
                x_i = X_train[i]
                y_i = y_train[i]

                # Compute scores for each class (including bias)
                scores = np.dot(self.w, x_i) + self.b

                # Compute the score for the correct class
                correct_class_score = scores[y_i]

                # Compute the hinge loss for each class
                loss_i = 0
                for c in range(self.n_class):
                    if c == y_i:
                        continue
                    margin = scores[c] - correct_class_score + self.delta
                    if margin > 0:
                        loss_i += margin
                        # Update weights for incorrect class
                        self.w[c] -= current_lr * x_i
                        self.b[c] -= current_lr
                        # Update weights for correct class
                        self.w[y_i] += current_lr * x_i
                        self.b[y_i] += current_lr

                total_loss += loss_i

            # Print loss every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / N}")

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

        # for each test data point, predict a value
        # use dot product,
        scores = np.dot(X_test, self.w.T) + self.b

        return np.argmax(scores, axis=1)

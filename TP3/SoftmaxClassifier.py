from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):
    """A softmax classifier"""

    def __init__(self, lr=0.1, alpha=100, n_epochs=1000, eps=1.0e-5, threshold=1.0e-10, regularization=True,
                 early_stopping=True):

        """
            self.lr : the learning rate for weights update during gradient descent
            self.alpha: the regularization coefficient
            self.n_epochs: the number of iterations
            self.eps: the threshold to keep probabilities in range [self.eps;1.-self.eps]
            self.regularization: Enables the regularization, help to prevent overfitting
            self.threshold: Used for early stopping, if the difference between losses during
                            two consecutive epochs is lower than self.threshold, then we stop the algorithm
            self.early_stopping: enables early stopping to prevent overfitting
        """

        self.lr = lr
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.eps = eps
        self.regularization = regularization
        self.threshold = threshold
        self.early_stopping = early_stopping

    """
        Public methods, can be called by the user
        To create a custom estimator in sklearn, we need to define the following methods:
        * fit
        * predict
        * predict_proba
        * fit_predict        
        * score
    """

    """
        In:
        X : the set of examples of shape nb_example * self.nb_features
        y: the target classes of shape nb_example *  1

        Do:
        Initialize model parameters: self.theta_
        Create X_bias i.e. add a column of 1. to X , for the bias term
        For each epoch
            compute the probabilities
            compute the loss
            compute the gradient
            update the weights
            store the loss
        Test for early stopping

        Out:
        self, in sklearn the fit method returns the object itself


    """

    def fit(self, X, y=None):
        prev_loss = np.inf
        self.losses_ = []

        self.nb_classes = len(np.unique(y))
        X_bias = self.add_bias(X)
        self.theta_ = np.random.normal(scale=0.3, size=(X.shape[1] + 1, self.nb_classes))

        for epoch in range(self.n_epochs):
            logits = np.dot(X_bias, self.theta_)
            probabilities = self._softmax(logits)

            loss = self._cost_function(probabilities, y)
            # Updates weights
            self.theta_ = self.theta_ - self._get_gradient(X_bias, y, probabilities)

            self.losses_.append(loss)

            if self.early_stopping and self.threshold > abs(loss - prev_loss):
                return self

            prev_loss = loss

        return self

    """
        In: 
        X without bias

        Do:
        Add bias term to X

        Out:
        X with bias
    """
    def add_bias(self, X):
        # Creates a Numpy array
        np_x = np.array(X)
        # Creates array with one more column than X
        X_bias = np.ones((np_x.shape[0], X.shape[1] + 1))
        # Sets the values so the first column has ones and the others correspond to X
        X_bias[:, 1:] = np_x

        return X_bias

    """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax

        Out:
        Predicted probabilities
    """

    def predict_proba(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        X_bias = self.add_bias(X)
        logits = np.dot(X_bias, self.theta_)
        probabilities = self._softmax(logits)

        return probabilities

    """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax
        Predict the classes

        Out:
        Predicted classes
    """

    def predict(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        X_bias = self.add_bias(X)
        logits = np.dot(X_bias, self.theta_)
        probabilities = self._softmax(logits)

        return np.argmax(probabilities, axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X, y)

    """
        In : 
        X set of examples (without bias term)
        y the true labels

        Do:
            predict probabilities for X
            Compute the log loss without the regularization term

        Out:
        log loss between prediction and true labels

    """

    def score(self, X, y, **kwargs):
        self.regularization = False
        probabilities = self.predict_proba(X)
        log_loss = self._cost_function(probabilities, y)

        return log_loss

    """
        Private methods, their names begin with an underscore
    """

    """
        In :
        y without one hot encoding
        probabilities computed with softmax

        Do:
        One-hot encode y
        Ensure that probabilities are not equal to either 0. or 1. using self.eps
        Compute log_loss
        If self.regularization, compute l2 regularization term
        Ensure that probabilities are not equal to either 0. or 1. using self.eps

        Out: 
        cost (real number)
    """

    def _cost_function(self, probabilities, y):
        m = probabilities.shape[0]
        # One hot encoding of y
        yohe = self._one_hot(y)
        # Replaces 0 probabilities by eps
        np.place(probabilities, probabilities == 0, self.eps)
        # Replaces 1 probabilities by 1 - eps
        np.place(probabilities, probabilities == 1, 1 - self.eps)

        log_loss = (-1 / m) * (np.sum(np.sum(yohe * np.log(probabilities), axis=1), axis=0))

        if self.regularization:
            theta2 = np.delete(self.theta_, 0, 0)
            log_loss = log_loss + self.alpha * np.sum(np.sum(np.power(theta2, 2), axis=1), axis=0) / m

        # Replaces 0 probabilities by eps
        np.place(probabilities, probabilities == 0, self.eps)
        # Replaces 1 probabilities by 1 - eps
        np.place(probabilities, probabilities == 1, 1 - self.eps)

        return log_loss


    """
        In :
        Target y: nb_examples * 1

        Do:
        One hot-encode y
        [1,1,2,3,1] --> [[1,0,0],
                         [1,0,0],
                         [0,1,0],
                         [0,0,1],
                         [1,0,0]]
        Out:
        y one-hot encoded
    """

    def _one_hot(self, y):
        # Creates a Numpy array
        np_y = np.array(y)
        # Training example count
        m = np_y.size
        # Different categories
        categories = np.unique(np_y)
        # Creates a zero matrix whose number of columns corresponds to the number of categories
        # and the number of rows corresponds to the numbers of training examples
        yohe = np.zeros((m, categories.size))
        # Formats array by replacing categories number by their index in categories array in order
        # not to exceed one_hot_array size during the next step
        for index in range(len(categories)):
            np.place(np_y, np_y == categories[index], index)

        # Sets ones to the concerned values without using loop directly
        yohe[np.arange(m), np_y] = 1

        return yohe

    """
        In: 
        nb_examples * self.nb_classes

        Do:
        Compute softmax on logits

        Out:
        Probabilities
    """

    def _softmax(self, z):
        somme = np.sum(np.exp(z), axis=1)
        return np.apply_along_axis(lambda x: np.exp(x) / somme, 0, z)


    """
        In:
        X with bias
        y without one hot encoding
        probabilities resulting of the softmax step

        Do:
        One-hot encode y
        Compute gradients
        If self.regularization add l2 regularization term

        Out:
        Gradient

    """

    def _get_gradient(self, X, y, probabilities):
        m = X.shape[0]
        # One hot encoding of y
        yohe = self._one_hot(y)
        # Computes costs function gradient
        gradient = self.lr * (np.dot(np.transpose(X), (probabilities - yohe)) / float(m))

        if self.regularization:
            theta2 = np.delete(self.theta_, 0, 0)
            gradient = gradient + self.alpha * np.sum(np.sum(np.power(theta2, 2), axis=1), axis=0) * self.theta_ / m

        return gradient
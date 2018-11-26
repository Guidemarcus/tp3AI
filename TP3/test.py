from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

def _one_hot(X):
    # Creates a Numpy array
    np_y = np.array(X)
    # Training example count
    m = np_y.size
    # Different categories
    categories = np.unique(np_y)
    # Creates a zero matrix whose number of columns corresponds to the number of categories
    # and the number of rows corresponds to the numbers of training examples
    one_hot_array = np.zeros((m, categories.size))
    # Formats array by replacing categories number by their index in categories array in order
    # not to exceed one_hot_array size during the next step
    for index in range(len(categories)):
        np.place(np_y, np_y == categories[index], index)

    # Sets ones to the concerned values without using loop directly
    one_hot_array[np.arange(m), np_y] = 1

    return one_hot_array

print(np.exp([[1,2, 3],[1,2, 3],[1,2, 3]]))
print(np.sum(np.exp([[1,2, 3],[1,2, 3],[1,2, 3]]), axis=1))
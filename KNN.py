import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, x, y):
        self.X_train = x
        self.Y_train = y

    def predict(self, X):
        prediction = [self.nearstK(i) for i in X]
        return np.array(prediction)

    def nearstK(self, new_point):
        distances = [euclidean_distance(new_point, x) for x in self.X_train]
        KN = np.argsort(distances)[:self.k]
        KN_lables = [self.Y_train[i] for i in KN]
        highest_frequency = Counter(KN_lables).most_common(1)
        return highest_frequency[0][0]

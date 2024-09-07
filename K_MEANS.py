import numpy as np


class KMeans:

    def __init__(self, k, max_iter):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.Wcss = 0

    def initialize(self, centroids):
        new_centroids = centroids.astype(float)
        self.centroids = new_centroids

    @staticmethod
    def per_point(self, point, dict_of_centroids):
        new_dict = {}
        for key in dict_of_centroids.keys():
            temp_point = dict_of_centroids[key]
            distance = np.linalg.norm(point - temp_point)
            new_dict[distance] = key
        list_of_distance = sorted(new_dict.keys())
        return new_dict[list_of_distance[0]]

    @staticmethod
    def find_center(self, array):
        center = []
        total = array.shape[1]
        for i in range(total):
            center += [np.mean(array[:, i])]
        return center

    def wcss(self):
        return self.Wcss

    def fit(self, X_train):
        dict_of_centroids = {}
        for i in range(self.centroids.shape[0]):
            dict_of_centroids[i+1] = self.centroids[i, :]
        total_iter = 0
        value_shape = X_train.shape
        cheak = False
        new_array = np.zeros(shape=(value_shape[0], 1))
        while total_iter <= self.max_iter and cheak is False:
            new_array = np.zeros(shape=(value_shape[0], 1))
            cheak = True
            total_iter += 1
            new_array = np.append(X_train, new_array, axis=1)
            new_shape = new_array.shape
            for i in range(value_shape[0]):
                point = new_array[i, :new_shape[1] - 1]
                new_array[i, new_shape[1] - 1] = KMeans.per_point(self, point, dict_of_centroids)
            list_of_keys = dict_of_centroids.keys()
            for i in range(len(list_of_keys)):
                temp_array = new_array[new_array[:, new_shape[1] - 1] == i + 1]
                new_center = KMeans.find_center(self, temp_array[:, :new_shape[1] - 1])
                if np.linalg.norm(np.array(dict_of_centroids[i+1]) - np.array(new_center)) != 0:
                    cheak = False
                dict_of_centroids[i + 1] = new_center
            for i in range(len(list_of_keys)):
                self.centroids[i, :] = dict_of_centroids[i + 1]
        summ = 0
        for i in range(X_train.shape[0]):
            key = int(new_array[i, new_array.shape[1] - 1])
            point = new_array[i, :new_array.shape[1] - 1]
            center = self.centroids[key - 1, :]
            temp = (np.linalg.norm(np.array(center) - np.array(point))) ** 2
            summ += temp
        self.Wcss = summ
        return dict_of_centroids

    def predict(self, X):
        dict_of_centroids = {}
        for i in range(self.centroids.shape[0]):
            dict_of_centroids[i+1] = self.centroids[i, :]
        new_array = np.zeros(shape=(X.shape[0], 1))
        new_array = np.append(X, new_array, axis=1)
        new_shape = new_array.shape
        for i in range(X.shape[0]):
            point = new_array[i, :new_shape[1] - 1]
            new_array[i, new_shape[1] - 1] = KMeans.per_point(self, point, dict_of_centroids)
        return np.array(new_array[:, new_shape[1] - 1], dtype=int)

import numpy as np


class KNN:
    def __init__(self, k: int) -> None:
        self.k = k
        self.value = None
        self.names = None

    def fit(self, x_train, y_train) -> None:
        self.value = x_train
        self.names = y_train

    @staticmethod
    def most_common_in_a_list(self, cheak_list):
        count = 0
        num = cheak_list[0]
        for i in cheak_list:
            cur_frc = cheak_list.count(i)
            if cur_frc > count:
                count = cur_frc
                num = i
        return num

    def predict(self, x_test):
        value_shape = self.value.shape ## rows = number of data sample
        numbers_of_samples = value_shape[0]
        first_value = self.names[0]
        result = np.array(range(x_test.shape[0]), dtype=type(first_value))
        for i in range(x_test.shape[0]):
            j = 0
            dict = {}
            point_1 = x_test[i, :]
            all_distances = []
            for j in range(numbers_of_samples):
                point_2 = self.value[j, :]
                distance = np.linalg.norm(point_1 - point_2)
                all_distances += [distance]
                if distance in dict.keys():
                    dict[distance].append(self.names[j])
                else:
                    dict[distance] = [self.names[j]]
            sorted_distance = sorted(all_distances)
            all_names = []
            befor = -1
            for n in range(len(sorted_distance)):
                if sorted_distance[n] != befor:
                    all_names += dict[sorted_distance[n]]
                befor = sorted_distance[n]
            temp = min(self.k, len(all_names))
            all_names = all_names[0:temp]
            name = KNN.most_common_in_a_list(self, all_names)
            result[i] = name
        return result

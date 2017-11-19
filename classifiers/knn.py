# knn.py
# Created by abdularis on 19/11/17

import numpy as np
import datautil


class KNearestNeighbors:
    """
        This is K-Nearest Neighbors which try to find the top K closest neighbors
    """

    def __init__(self):
        self.x_training = None
        self.y_training = None

    def train(self, x, y):
        """In K-Nearest Neighbors there's no training step
           because prediction process is just comparing all
           new data with these training data set
           """
        self.x_training = x
        self.y_training = y

    def predict(self, x_new, k=1):
        num_loop = x_new.shape[0]

        # prediction results, which are labels for all new data (x)
        y_predictions = np.zeros(num_loop, dtype=self.y_training.dtype)

        for i in range(num_loop):

            print("Predicting: {}%".format(i * 100 / num_loop))

            # using L1/Manhattan distance
            # you can also use L2/Euclidean distance
            distances = np.sum(np.abs(self.x_training - x_new[i, :]), axis=1)

            # get the k index of the closest neighbors
            closest_idx = np.argsort(distances)[:k]

            # classes extracted from closest classes index
            classes = [self.y_training[idx] for idx in closest_idx]

            # find which has the most occurrence in the classes & assign it to the y_predictions for result
            counts = np.bincount(classes)
            y_predictions[i] = np.argmax(counts)

        return y_predictions


Xtr, Ytr, Xte, Yte = datautil.load_cifar10('../dataset')


# Xtr_rows becomes 50000 x 3072
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)

# Xte_rows becomes 10000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

knn_clasifier = KNearestNeighbors()
knn_clasifier.train(Xtr_rows, Ytr)

Yte_predict = knn_clasifier.predict(Xte_rows, k=3)

print('Accuracy: %f' % np.mean(Yte_predict == Yte))

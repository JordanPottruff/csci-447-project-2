
from src.algorithms.knn import KNN


class CondensedKNN(KNN):

    def __init__(self, training_data, k):
        super().__init__(training_data, k)
        self.condense_training_data()

    def condense_training_data(self):
        # No return value -- method changes self.training_data variable.
        'empty'

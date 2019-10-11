
from src.algorithms.knn import KNN
import src.util as util
import src.datasets.data_set as ds
import src.loss as loss


class EditedKNN(KNN):
    def __init__(self, training_data, k):
        super().__init__(training_data, k)
        self.training_data = training_data.copy()
        self.find_edited_data()
        self.edited_data_size = self.get_edited_data_size()

    # Updates the edit_training_data variable to the edited data_set, that is the data_set with unnecessary
    # vectors removed
    def find_edited_data(self):
        counter = 0
        length = len(self.training_data.data)
        while True:
            if counter > length:
                break
            counter += 1

            # prev_accuracy = self.get_validation_accuracy()
            example = self.training_data.data.pop(0)
            # new_accuracy = self.get_validation_accuracy()
            if util.get_highest_class(self.run(example)) != example[self.training_data.class_col]:
                self.training_data.data.append(example)
            else:
                length -= 1
                counter = 0

    def get_validation_accuracy(self):
        results = []
        for example in self.validation_data.data:
            results.append({"actual": self.run(example), "expected": example[self.validation_data.class_col]})
        return loss.calc_hinge(results)

    # Returns the length of the edited_data_set
    def get_edited_data_size(self):
        return len(self.training_data.get_data())

    # Returns the list of removed data set
    def get_removed_data_set(self):
        return self.removed_data_set







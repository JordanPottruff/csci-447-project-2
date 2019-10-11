
from src.algorithms.knn import KNN
import src.util as util
import src.datasets.data_set as ds
import src.loss as loss


class EditedKNN(KNN):
    def __init__(self, training_data, k):
        super().__init__(training_data, k)
        partitions = training_data.partition(.20)
        self.training_data = partitions[1]
        self.validation_data = partitions[0]
        self.removed_data_set = []
        self.find_edited_data()
        self.edited_data_size = self.get_edited_data_size()

    # Updates the edit_training_data variable to the edited data_set, that is the data_set with unnecessary
    # vectors removed
    def find_edited_data(self):
        while True:
            previous_accuracy = self.validate_accuracy()

            example = self.training_data.data.pop(0)
            if util.get_highest_class(self.run(example)) != example[self.training_data.class_col]:
                self.training_data.data.append(example)
                continue

            new_accuracy = self.validate_accuracy()
            if new_accuracy < previous_accuracy:
                break

    def validate_accuracy(self):
        return loss.calc_accuracy([{"actual": self.run(example), "expected": example[self.validation_data.class_col]} for example in self.validation_data.data]) / len(self.validation_data.data)

    # Returns the length of the edited_data_set
    def get_edited_data_size(self):
        return len(self.training_data.get_data())

    # Returns the list of removed data set
    def get_removed_data_set(self):
        return self.removed_data_set







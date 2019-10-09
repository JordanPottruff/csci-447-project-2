
from src.algorithms.knn import KNN
import src.util as util
import src.datasets.data_set as ds


class EditedKNN(KNN):
    def __init__(self, training_data, k):
        super().__init__(training_data, k)
        self.training_data = training_data
        self.training_data = self.find_edited_data()
        self.edited_data_size = self.get_edited_data_size()

    # Updates the edit_training_data variable to the edited data_set, that is the data_set with unnecessary
    # vectors removed
    def find_edited_data(self) -> list:
        edited_data_set = self.training_data.get_data()
        # print("Original Data_Set Length: " + str(len(edited_data_set)))
        for index, data in enumerate(self.training_data.get_data()):
            if util.get_highest_class(self.run(data)) == data[self.training_data.class_col]:
                edited_data_set.pop(index)
        self.edited_data_size = len(edited_data_set)
        # print("Edited Data_Set Length: " + str(len(edited_data_set)))
        return ds.DataSet(edited_data_set, self.training_data.class_col, self.training_data.attr_cols)

    # Returns the length of the edited_data_set
    def get_edited_data_size(self):
        return len(self.training_data.get_data())






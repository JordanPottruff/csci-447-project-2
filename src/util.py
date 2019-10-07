# Here we can store functions that are useful across all algorithms.


# Calculates the class distribution of a 2D list of data. The distribution is stored in a dictionary that maps each
# class to the number of times the class is present in the data.
def calculate_class_distribution(data, class_col):
    freq = {}
    for obs in data:
        class_val = obs[class_col]
        if class_val in freq:
            freq[class_val] += 1
        else:
            freq[class_val] = 1
    return freq

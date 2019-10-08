
import src.util as util
import src.datasets.data_set as ds
import src.algorithms.k_means as kmeans
import src.algorithms.knn as k_nn


def get_abalone_data():
    data = util.read_file("../data/abalone.data")
    abalone_data = ds.DataSet(data, 8, list(range(0, 8)))
    numeric_columns = list(range(1, 8))
    # Convert attribute columns to floats
    abalone_data.convert_to_float(numeric_columns)
    # Normalize values
    abalone_data.normalize_z_score(numeric_columns)
    return abalone_data


def get_car_data():
    data = util.read_file("../data/car.data")
    car_data = ds.DataSet(data, 6, list(range(0, 6)))
    # Convert attribute columns to numeric scheme
    car_data.convert_attribute(0, {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    car_data.convert_attribute(1, {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    car_data.convert_attribute(2, {'2': 2, '3': 3, '4': 4, '5more': 5})
    car_data.convert_attribute(3, {'2': 2, '4': 4, 'more': 5})
    car_data.convert_attribute(4, {'small': 0, 'med': 1, 'big': 2})
    car_data.convert_attribute(5, {'low': 0, 'med': 1, 'high': 2})
    numeric_columns = list(range(0, 6))
    # Normalize values.
    car_data.normalize_z_score(numeric_columns)
    return car_data


def get_forest_fires_data():
    data = util.read_file("../data/forestfires.data")
    forest_fires_data = ds.DataSet(data, 12, list(range(0, 12)))
    numeric_columns = [0, 1] + list(range(4, 13))
    # Remove the first line, which is the header info.
    forest_fires_data.remove_header()
    # Convert applicable columns to floats, including the class column.
    forest_fires_data.convert_to_float(numeric_columns)
    # Normalize values.
    forest_fires_data.normalize_z_score(numeric_columns)
    return forest_fires_data


def get_machine_data():
    data = util.read_file("../data/machine.data")
    # There is another final column but we probably want to exclude it.
    machine_data = ds.DataSet(data, 8, list(range(0, 8)))
    numeric_columns = list(range(2, 9))
    # Convert all columns except the first two to floats, including the class column.
    machine_data.convert_to_float(numeric_columns)
    # Normalize values.
    machine_data.normalize_z_score(numeric_columns)
    return machine_data


def run_k_means(data_set, k):
    print("-------")
    print("K-MEANS")
    print("-------")
    print("DataSet: " + data_set.filename)
    print("Initializing k-means: finding centroids...")
    km = kmeans.KMeans(data_set, k)
    print("Running k-means: ")


def main():
    abalone_data = get_abalone_data()
    car_data = get_car_data()
    forest_fires_data = get_forest_fires_data()
    machine_data = get_machine_data()

    # km = kmeans.KMeans(machine_data, 2)
    # print(km.centroids)

    knn = k_nn.KNN(abalone_data, 1)
    print(knn.calc_euclidean_distance())

    knn.find_k_smallest()




main()
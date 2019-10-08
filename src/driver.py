
import src.util as util
import src.loss as loss
import src.datasets.data_set as ds
import src.algorithms.k_means as kmeans
import src.algorithms.knn as k_nn


ABALONE_DATA_FILE = "../data/abalone.data"
CAR_DATA_FILE = "../data/car.data"
FOREST_FIRE_DATA_FILE = "../data/forestfires.data"
MACHINE_DATA_FILE = "../data/machine.data"


def get_abalone_data():
    data = util.read_file(ABALONE_DATA_FILE)
    abalone_data = ds.DataSet(data, 8, list(range(0, 8)), ABALONE_DATA_FILE)
    numeric_columns = list(range(1, 8))
    # Convert attribute columns to floats
    abalone_data.convert_to_float(numeric_columns)
    # Normalize values
    abalone_data.normalize_z_score(numeric_columns)
    return abalone_data


def get_car_data():
    data = util.read_file(CAR_DATA_FILE)
    car_data = ds.DataSet(data, 6, list(range(0, 6)), CAR_DATA_FILE)
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
    data = util.read_file(FOREST_FIRE_DATA_FILE)
    forest_fires_data = ds.DataSet(data, 12, list(range(0, 12)), FOREST_FIRE_DATA_FILE)
    numeric_columns = [0, 1] + list(range(4, 13))
    # Remove the first line, which is the header info.
    forest_fires_data.remove_header()
    # Convert applicable columns to floats, including the class column.
    forest_fires_data.convert_to_float([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # Normalize values.
    forest_fires_data.normalize_z_score([0, 1, 4, 5, 6, 7, 8, 9, 10, 11])
    return forest_fires_data


def get_machine_data():
    data = util.read_file(MACHINE_DATA_FILE)
    # There is another final column but we probably want to exclude it.
    machine_data = ds.DataSet(data, 8, list(range(0, 8)), MACHINE_DATA_FILE)
    # Convert all columns except the first two to floats, including the class column.
    machine_data.convert_to_float(list(range(2, 8)))
    # Normalize values.
    machine_data.normalize_z_score(list(range(2, 8)))
    return machine_data


def run_k_means(data_set, k):
    print("-------")
    print("K-MEANS")
    print("-------")
    print("Data Set: " + data_set.filename)
    folds = data_set.validation_folds(10)

    for i, fold in enumerate(folds):
        print("Fold " + str(i + 1) + ": ")
        test = fold['test']
        train = fold['train']
        km = kmeans.KMeans(train, k)
        print(" * distortion = " + str(km.distortion))

        results = []
        for obs in test.data:
            result = {"expected": obs[data_set.class_col], "actual": km.run(obs)}
            results.append(result)

        print(" * accuracy = " + str(loss.calc_accuracy(results)))


def main():
    abalone_data = get_abalone_data()
    car_data = get_car_data()
    forest_fires_data = get_forest_fires_data()
    machine_data = get_machine_data()

    run_k_means(abalone_data, 20)

main()
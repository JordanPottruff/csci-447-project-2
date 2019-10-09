
import src.util as util
import src.loss as loss
import src.datasets.data_set as ds
import src.algorithms.k_means as kmeans
import src.algorithms.knn as k_nn
import src.algorithms.edited_knn as e_nn


ABALONE_DATA_FILE = "../data/abalone.data"
CAR_DATA_FILE = "../data/car.data"
FOREST_FIRE_DATA_FILE = "../data/forestfires.data"
MACHINE_DATA_FILE = "../data/machine.data"
SEGMENTATION_DATA_FILE = "../data/segmentation.data"
WINE_DATA_FILE = "../data/wine.data"


def get_abalone_data():
    data = util.read_file(ABALONE_DATA_FILE)
    abalone_data = ds.DataSet(data, 8, list(range(0, 8)), ABALONE_DATA_FILE)
    numeric_columns = list(range(1, 8))
    # Convert attribute columns to floats
    abalone_data.convert_to_float(numeric_columns)
    # Normalize values
    abalone_data.normalize_z_score(numeric_columns)
    # Randomly shuffle values.
    abalone_data.shuffle()
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
    # Randomly shuffle values.
    car_data.shuffle()
    return car_data


def get_forest_fires_data():
    data = util.read_file(FOREST_FIRE_DATA_FILE)
    forest_fires_data = ds.DataSet(data, 12, list(range(0, 12)), FOREST_FIRE_DATA_FILE)
    numeric_columns = [0, 1] + list(range(4, 13))
    # Remove the first line, which is the header info.
    forest_fires_data.remove_header(1)
    # Convert applicable columns to floats, including the class column.
    forest_fires_data.convert_to_float([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # Normalize values.
    forest_fires_data.normalize_z_score([0, 1, 4, 5, 6, 7, 8, 9, 10, 11])
    # Randomly shuffle values.
    forest_fires_data.shuffle()
    return forest_fires_data


def get_machine_data():
    data = util.read_file(MACHINE_DATA_FILE)
    # There is another final column but we probably want to exclude it.
    machine_data = ds.DataSet(data, 8, list(range(0, 8)), MACHINE_DATA_FILE)
    # Convert all columns except the first two to floats, including the class column.
    machine_data.convert_to_float(list(range(2, 8)))
    # Normalize values.
    machine_data.normalize_z_score(list(range(2, 8)))
    # Randomly shuffle values.
    machine_data.shuffle()
    return machine_data


def get_segmentation_data():
    data = util.read_file(SEGMENTATION_DATA_FILE)
    segmentation_data = ds.DataSet(data, 0, list(range(1, 20)), SEGMENTATION_DATA_FILE)
    # Remove the first 5 lines, which is reserved for the header.
    segmentation_data.remove_header(5)
    # Convert all attribute columns to numeric values.
    segmentation_data.convert_to_float(list(range(1, 20)))
    # Normalize values.
    segmentation_data.normalize_z_score(list(range(1, 20)))
    # Randomly shuffle values.
    segmentation_data.shuffle()
    return segmentation_data


def get_wine_data():
    data = util.read_file(WINE_DATA_FILE)
    wine_data = ds.DataSet(data, 0, list(range(1, 14)), WINE_DATA_FILE)
    # Convert all attribute columns to numeric values.
    wine_data.convert_to_float(list(range(1, 14)))
    # Normalize values.
    wine_data.normalize_z_score(list(range(1, 14)))
    # Randomly shuffle values.
    wine_data.shuffle()
    return wine_data


def run_k_means(data_set, k):
    print("-------")
    print("K-MEANS")
    print("-------")
    print("Data Set: " + data_set.filename)
    folds = data_set.validation_folds(10)
    print()
    print("10-Fold Cross Validation:")

    avg_accuracy = 0
    avg_hinge = 0
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

        accuracy = loss.calc_accuracy(results)
        hinge = loss.calc_hinge(results)
        print(" * accuracy = " + str(accuracy))
        print(" * hinge loss = " + str(hinge))
        avg_accuracy += accuracy / len(folds)
        avg_hinge += hinge / len(folds)
    print("")
    print("Final Results: ")
    print(" * avg accuracy = " + str(avg_accuracy))
    print(" * avg hinge = " + str(avg_hinge))
    print()

def run_knn(data_set, k):
    print("-------")
    print("KNN")
    print("-------")
    print("Data Set: " + data_set.filename)
    folds = data_set.validation_folds(10)
    print()
    print("10-Fold Cross Validation:")

    avg_accuracy = 0
    avg_hinge = 0
    for i, fold in enumerate(folds):
        print("Fold " + str(i + 1) + ": ")
        test = fold['test']
        train = fold['train']
        knn = k_nn.KNN(train, k)

        # print(" * distortion = " + str(knn.distortion))

        results = []
        hc = []
        for obs in test.data:
            result = {"expected": obs[data_set.class_col], "actual": knn.run(obs)}
            results.append(result)

        accuracy = loss.calc_accuracy(results)
        hinge = loss.calc_hinge(results)
        print(" * accuracy = " + str(accuracy))
        print(" * hinge loss = " + str(hinge))
        avg_accuracy += accuracy / len(folds)
        avg_hinge += hinge / len(folds)
    print("")
    print("Final Results: ")
    print(" * avg accuracy = " + str(avg_accuracy))
    print(" * avg hinge = " + str(avg_hinge))
    print()

def run_enn(data_set, k):
    # enn = e_nn.EditedKNN(data_set, k)
    print("-------")
    print("KNN")
    print("-------")
    print("Data Set: " + data_set.filename)
    folds = data_set.validation_folds(10)
    print()
    print("10-Fold Cross Validation:")

    avg_accuracy = 0
    avg_hinge = 0
    for i, fold in enumerate(folds):
        print("Fold " + str(i + 1) + ": ")
        test = fold['test']
        train = fold['train']
        enn = e_nn.EditedKNN(train, k)

        # print(" * distortion = " + str(knn.distortion))

        results = []
        for obs in test.data:
            result = {"expected": obs[data_set.class_col], "actual": enn.run(obs)}
            results.append(result)

        accuracy = loss.calc_accuracy(results)
        hinge = loss.calc_hinge(results)
        print(" * accuracy = " + str(accuracy))
        print(" * hinge loss = " + str(hinge))
        avg_accuracy += accuracy / len(folds)
        avg_hinge += hinge / len(folds)
    print("")
    print("Final Results: ")
    print(" * avg accuracy = " + str(avg_accuracy))
    print(" * avg hinge = " + str(avg_hinge))
    print()


def test_knn(data_set, k):
    knn = k_nn.KNN(data_set, k)

    results = []
    total = len(data_set.data) // 2
    count = 0
    for obs in data_set.data[:total]:
        results.append({"expected": obs[data_set.class_col], "actual": knn.run(obs)})
        if count % 100 == 0:
            print(str(count*100 / total) + "%...")
        count += 1

    print("Test KNN")
    print("Accuracy: " + str(loss.calc_accuracy(results)))


def main():
    # Open data sets
    """Classification"""
    abalone_data = get_abalone_data()
    car_data = get_car_data()
    segmentation_data = get_segmentation_data()

    """Regression"""
    forest_fires_data = get_forest_fires_data()
    machine_data = get_machine_data()
    wine_data = get_wine_data()

    # Run k means algorithm
    # TODO: replace these k's with the size of the edited KNN training set.
    # run_k_means(abalone_data, 20)
    # run_k_means(car_data, 20)
    # run_k_means(segmentation_data, 20)
    # km = kmeans.KMeans(machine_data, 2)
    # print(km.centroids)

    # Run knn algorithm -----------------------------------------------
    # run_knn(abalone_data, 3)
    # run_knn(car_data, 3)
    # run_knn(segmentation_data, 3)

    # Run ENN algorithm -----------------------------------------------
    # run_enn(abalone_data, 3)
    # run_enn(car_data, 3)
    # run_enn(segmentation_data, 3)




main()

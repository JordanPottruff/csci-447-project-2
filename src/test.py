# This file can be used for testing out simplified data sets.

import src.util as util
import src.datasets.data_set as ds
import src.algorithms.k_means as kmeans
import src.algorithms.pam_nn as pamnn
import src.algorithms.knn as k_nn
import src.algorithms.edited_knn as e_knn
import src.algorithms.condensed_knn as c_knn

THREE_CLUSTERS_DATA_FILE = "../data/test/three_clusters.data"
CAR_DATA_FILE = "../data/car.data"


def get_three_clusters_data():
    data = util.read_file(THREE_CLUSTERS_DATA_FILE)
    three_clusters_data = ds.DataSet(data, 2, [0,1], THREE_CLUSTERS_DATA_FILE)
    three_clusters_data.convert_to_float([0, 1])
    #three_clusters_data.normalize_z_score([0, 1])
    return three_clusters_data

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
    # car_data.normalize_z_score(numeric_columns)
    # Randomly shuffle values.
    car_data.shuffle()
    return car_data

# CLASSIFICATION


def test_kmeans():
    three_clusters_data = get_three_clusters_data()

    km = kmeans.KMeans(three_clusters_data, 3)
    # For verification, each class should have its own cluster:
    print("Clusters from kmeans:")
    for count, cluster_classes in enumerate(km.cluster_classes, 1):
        print(str(count) + ": " + str(cluster_classes))


def test_pam():
    three_clusters_data = get_three_clusters_data()

    pam = pamnn.PamNN(three_clusters_data, 3)
    print("Clusters from PAM:")
    for count, cluster_classes in enumerate(pam.cluster_classes, 1):
        print(str(count) + ": " + str(cluster_classes) + ", " + str(pam.medoids[count-1]))


def test_knn():
    test_knn_data = get_three_clusters_data()
    test = [4,3,'A']
    # CAUTION: test variable is not normalized but the data is
    kn = k_nn.KNN(test_knn_data, 22)
    print(kn.run(test))

# Neighbors, Clusters, Whether a point was added or removed

def display_enn(original_data:list, edited_data:list, removed_data:list):
    print()
    print("=========================================")
    print("Running Edited KNN Algorithm")
    print("=========================================")
    print("ORIGINAL DATA")
    for data in original_data:
        print(data)
    print("EDITED DATA")
    for da in edited_data:
       print(da)
    print("REMOVED DATA")
    for d in removed_data:
        print(d)
    # print(len(original_data) - len(removed_data))


def display_nearest_neighbors(prob_map, k_closest, test):
    print("=========================================")
    print("Running K-Nearest Neighbors Algorithm")
    print("=========================================")
    print()
    print("|" + "CLASS DISTRIBUTION OF K-NN VECTORS".center(36)+"|")
    print("|------------------------------------|")
    print("|     CLASS     |    PROBABILITY     |")
    print("|---------------|--------------------|")
    for classes, probability in prob_map.items():
        print("|" + str(classes).center(15) + "|" + str(round(probability, 4)).center(20) + "|")
    print("|------------------------------------|")
    print()
    print("K = " + str(len(k_closest))+" CLOSEST NEIGHBORS OF " + str(test))
    for neighbors in k_closest:
        print(("  " + str(neighbors)).center(40))


def display_k_means(centroids, clusters):
    print("=======================================")
    print("Running K-Means Algorithms")
    print("=======================================")
    print()
    print("K = " + str(len(centroids)) + " CENTROIDS VECTORS   ")
    print("--------------------------------------")
    for index, centroid in enumerate(centroids):
        centroid = [round(centroid[i], 4) if centroid[i] is not None else 0 for i in range(len(centroid))]
        print("| " + str(index+1).rjust(4) + " | " + str(centroid).ljust(40))
    print()
    print("--------------------------------------")
    print("CLASS PROBABILITY OF EACH CLUSTER   ")
    print("--------------------------------------")
    for index, cluster in enumerate(clusters):
        for cluster_class, prob in cluster.items():
            cluster[cluster_class] = round(prob, 4)
        print("| " + str(index+1).rjust(4) + " |  " + str(cluster))
        # TODO: Is this an appropriate display for the video
        # print(str(index + 1) + " CLUSTER:")
        # print("     |------------------------------------|".rjust(50))
        # for cluster_class, prob in cluster.items():
        #     # for classes, probability in prob_map.items():
        #     print(("     |" + str(cluster_class).center(15) + "|" + str(round(prob, 4)).center(20) + "|").rjust(50))
        # print("     |------------------------------------|".rjust(50))


def main():
    # Test environment for the video

    data = get_three_clusters_data()

    knn = k_nn.KNN(data, 20)
    enn = e_knn.EditedKNN(data, 5)
    # cnn = c_knn.CondensedKNN(data, 20)
    k_means = kmeans.KMeans(data, 5)

    # CUSTOMIZED: TEST VECTOR FOR THREE CLUSTER DATA
    test = data.get_data()[0]

    prob_map = knn.run(test)
    k_closest = knn.find_closest_neighbors(test)

    centroids = k_means.centroids
    clusters = k_means.cluster_classes

    # TODO: Should we make the ENN object store the original data along with the edited data? We currently are simply
    #   updating the training data. The original data is erased when ENN is called.
    enn_removed_data = enn.get_removed_data_set()
    enn_training_data = enn.training_data.get_data()


    display_nearest_neighbors(prob_map, k_closest, test)
    print()
    display_k_means(centroids, clusters)
    display_enn(get_three_clusters_data().get_data(), enn_training_data, enn_removed_data)



    # for fold in data.validation_folds(10):
    #     for row in fold['train'].data:
    #         print(row)
    #     for row in fold['test'].data:
    #         print(row)
    #     print()
    #     print()
    # test_pam()



main()
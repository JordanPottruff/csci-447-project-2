# This file can be used for testing out simplified data sets.

import src.util as util
import src.datasets.data_set as ds
import src.algorithms.k_means as kmeans
import src.algorithms.pam_nn as pamnn
import src.algorithms.knn as k_nn

THREE_CLUSTERS_DATA_FILE = "../data/test/three_clusters.data"


def get_three_clusters_data():
    data = util.read_file(THREE_CLUSTERS_DATA_FILE)
    three_clusters_data = ds.DataSet(data, 2, [0,1], THREE_CLUSTERS_DATA_FILE)
    three_clusters_data.convert_to_float([0, 1])
    #three_clusters_data.normalize_z_score([0, 1])
    return three_clusters_data

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

# Neighbors, Clusters, Whether a point was added

def display_class_probability(classes:dict):
    pass


def display_nearest_neighbors(prob_map, k_closest, test):
    print("|  CLASS  |    PROBABILITY    |")
    print("|---------|-------------------|")
    for classes, probability in prob_map.items():
        print("|" + str(classes).center(9) + "|" + str(round(probability, 4)).center(19) + "|")

    print()
    print("K = " + str(len(k_closest))+" CLOSEST NEIGHBORS OF " + str(test))
    print("---------------------------------------------")
    for neighbors in k_closest:
        print(neighbors)

def display_k_means(centroids, clusters):
    print("CENTROIDS")
    for cent_points in centroids:
        print(cent_points)
    print("CLUSTERS")
    for clust in clusters:
        print(clust)






def main():
    data = get_three_clusters_data()
    knn = k_nn.KNN(data, 5)
    k_means = kmeans.KMeans(data, 5)

    # CUSTOMIZED: TEST VECTOR FOR THREE CLUSTER DATA
    test = data.get_data()[0]

    # CLASSIFICATION: TEST VECTOR FOR CAR DATA

    # REGRESSION: TEST VECTOR FOR WINE DATA

    prob_map = knn.run(test)
    k_closest = knn.find_closest_neighbors(test)

    display_nearest_neighbors(prob_map, k_closest, test)

    centroids = k_means.centroids
    clusters = k_means.cluster_classes

    display_k_means(centroids, clusters)



    # for fold in data.validation_folds(10):
    #     for row in fold['train'].data:
    #         print(row)
    #     for row in fold['test'].data:
    #         print(row)
    #     print()
    #     print()
    # test_pam()



main()
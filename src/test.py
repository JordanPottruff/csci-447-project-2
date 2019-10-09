# This file can be used for testing out simplified data sets.

import src.util as util
import src.datasets.data_set as ds
import src.algorithms.k_means as kmeans
import src.algorithms.pam_nn as pamnn
import src.algorithms.knn as knn

THREE_CLUSTERS_DATA_FILE = "../data/test/three_clusters.data"

def get_three_clusters_data():
    data = util.read_file(THREE_CLUSTERS_DATA_FILE)
    three_clusters_data = ds.DataSet(data, 2, [0,1], THREE_CLUSTERS_DATA_FILE)
    three_clusters_data.convert_to_float([0, 1])
    #three_clusters_data.normalize_z_score([0, 1])
    return three_clusters_data


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
    kn = knn.KNN(test_knn_data, 22)
    print(kn.run(test))





def main():
    data = get_three_clusters_data()
    for fold in data.validation_folds(10):
        for row in fold['train'].data:
            print(row)
        for row in fold['test'].data:
            print(row)
        print()
        print()
    #test_pam()


main()
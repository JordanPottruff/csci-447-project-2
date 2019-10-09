
import src.util as util
import src.loss as loss
import src.datasets.data_set as ds
import src.algorithms.k_means as kmeans
import src.algorithms.pam_nn as pamnn
import src.algorithms.knn as k_nn
import src.algorithms.edited_knn as e_nn
import src.algorithms.condensed_knn as ck_nn


def run_classification(alg_class, data_set, k_values):
    print("-----------------------------------------")
    print("CLASSIFICATION USING " + alg_class.__name__)
    print("Data: " + data_set.filename)
    print(" * N = " + str(len(data_set.data)))

    folds = data_set.validation_folds(10)
    for i, k in enumerate(k_values):
        print("[" + str(i+1) + "] k=" + str(k) + " using 10-fold CV")
        avg_accuracy = 0
        avg_hinge_loss = 0
        print(" * Folds Complete: ", end='', flush=True)
        for fold_i, fold in enumerate(folds):
            test = fold['test']
            train = fold['train']
            alg = alg_class(train, k)

            results = []
            for obs in test.data:
                result = {"expected": obs[data_set.class_col], "actual": alg.run(obs)}
                results.append(result)

            accuracy = loss.calc_accuracy(results)
            hinge_loss = loss.calc_hinge(results)
            avg_accuracy += accuracy / len(folds)
            avg_hinge_loss += hinge_loss / len(folds)
            print(fold_i+1, end='', flush=True)
            if fold_i == len(folds)-1:
                print()
            else:
                print(", ", end='', flush=True)

        print(" * Results: ")
        print("   - Avg accuracy = " + str(avg_accuracy))
        print("   - Avg hinge loss = " + str(avg_hinge_loss))
        print()


def run_regression(alg_class, data_set, k_values):
    print("-----------------------------------------")
    print("REGRESSION USING " + alg_class.__name__)
    print("Data: " + data_set.filename)
    print(" * N = " + str(len(data_set.data)))

    folds = data_set.validation_folds(10)
    for i, k in enumerate(k_values):
        print("[" + str(i+1) + "] k=" + str(k) + " using 10-fold CV")
        avg_mse = 0
        avg_los_cosh_loss = 0
        print(" * Folds Complete: ", end='', flush=True)
        for fold_i, fold in enumerate(folds):
            test = fold['test']
            train = fold['train']
            alg = alg_class(train, k)

            results = []
            for obs in test.data:
                result = {"expected": obs[data_set.class_col], "actual": alg.run(obs)}
                results.append(result)

            mse = loss.calc_mse(results)
            log_cosh_loss = loss.calc_log_cosh(results)
            avg_mse += mse / len(folds)
            avg_los_cosh_loss += log_cosh_loss / len(folds)
            print(fold_i + 1, end='', flush=True)
            if fold_i == len(folds) - 1:
                print()
            else:
                print(", ", end='', flush=True)
        print(" * Results: ")
        print("   - Avg mean squared error = " + str(avg_mse))
        print("   - Avg los cosh loss = " + str(avg_los_cosh_loss))
        print()


def main():
    # Classification data sets
    abalone_data = ds.get_abalone_data()
    car_data = ds.get_car_data()
    segmentation_data = ds.get_segmentation_data()
    # Regression data sets
    forest_fires_data = ds.get_forest_fires_data()
    machine_data = ds.get_machine_data()
    wine_data = ds.get_wine_data()

    # Classification analysis:
    run_classification(k_nn.KNN, abalone_data, [10, 30, 50])
    run_classification(k_nn.KNN, car_data, [10, 30, 50])
    run_classification(k_nn.KNN, segmentation_data, [10, 30, 50])

    # Regression analysis:
    run_regression(k_nn.KNN, machine_data, [5, 10, 15])
    run_regression(k_nn.KNN, forest_fires_data, [5, 10, 15])
    run_regression(k_nn.KNN, wine_data, [5, 10, 15])


main()

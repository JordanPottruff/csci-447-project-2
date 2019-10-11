import math


def calc_accuracy(results):
    correct = 0
    for result in results:
        expected_cls = result['expected']
        actual_dist = result['actual']
        highest_cls = get_highest_class(actual_dist)
        if expected_cls == highest_cls:
            correct += 1
    return correct / len(results)


def calc_hinge(results):
    hinge_sum = 0
    for result in results:
        expected_cls = result['expected']
        actual_classes = result['actual']
        expected_cls_prob = 0
        if expected_cls in actual_classes:
            expected_cls_prob = actual_classes[expected_cls]

        for cls in actual_classes:
            if cls == expected_cls:
                continue
            hinge_sum += max(0, actual_classes[cls] - expected_cls_prob + 1)
    return hinge_sum / len(results)


# Loss function that is more resilient to large residuals than straight forward mean square error.
def calc_huber_loss(results):
    huber_loss_sum = 0
    # For each result generated from test set
    mean, sd = calc_distribution(results)
    for result in results:
        expected_val = result['expected'] # Use log to make data difference smaller
        expected_z_score = (expected_val - mean) / sd
        actual_val = get_expected_value(result['actual']) # Use log to make data difference smaller
        actual_z_score = (actual_val - mean) / sd

        # If the values are more than one standard deviation apart, use MAE. Otherwise, MSE.
        hyper_param = 1
        if abs(actual_z_score - expected_z_score) <= hyper_param:
            huber_loss_sum += (actual_val - expected_val)**2
        else:
            huber_loss_sum += abs(actual_val - expected_val)
    return huber_loss_sum / len(results)
        

def calc_distribution(results):
    mean = 0
    for result in results:
        mean += result['expected'] / len(results)

    standard_deviation = 0
    for result in results:
        standard_deviation = ((result['expected'] - mean)**2) / len(results)
    standard_deviation = math.sqrt(standard_deviation)

    return mean, standard_deviation


def calc_rmse(results):
    rmse_sum = 0
    for result in results:
        expected_val = result['expected']
        actual_val = get_expected_value(result['actual'])
        rmse_sum += (expected_val - actual_val)**2
    return math.sqrt(rmse_sum / len(results))


def get_expected_value(distribution):
    weighted_sum = 0
    for value in distribution:
        weighted_sum += value * distribution[value]
    return weighted_sum


def get_highest_class(classes):
    highest_cls = None
    for cls in classes:
        if highest_cls is None or classes[cls] > classes[highest_cls]:
            highest_cls = cls
    return highest_cls

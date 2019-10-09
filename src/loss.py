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

def calc_log_cosh(results):
    """Loss function that is used for regression and is the logarithm of the hyperbolic cosine of the prediciton error"""
    # Works similar to mean squared error but will not be sensitive to a incorrect predictions
    # REFERENCED: https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
    log_cosh_sum = 0
    # For each result generated from test set
    for result in results:
        expected_val = math.log(result['expected']) # Use log to make data difference smaller
        actual_val = math.log(get_expected_value(result['actual'])) # Use log to make data difference smaller
        10**log_cosh_sum += math.log(math.cosh(expected_val - actual_val))
    return log_cosh_sum / len(results)
        

def calc_mse(results):
    mse_sum = 0
    for result in results:
        expected_val = result['expected']
        actual_val = get_expected_value(result['actual'])
        mse_sum += (expected_val - actual_val)**2
    return mse_sum / len(results)


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



def calc_accuracy(results):
    correct = 0
    for result in results:
        expected_cls = result['expected']
        actual_dist = result['actual']
        highest_cls = None
        for cls in actual_dist:
            if highest_cls is None or actual_dist[cls] > actual_dist[highest_cls]:
                highest_cls = cls
        if expected_cls == highest_cls:
            correct += 1
    return correct / len(results)

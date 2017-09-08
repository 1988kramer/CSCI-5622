import argparse
import random
from collections import namedtuple
import numpy as np
from knn import Knearest, Numbers

random.seed(20170830)


# READ THIS FIRST
# In n-fold cross validation, all the instances are split into n folds
# of equal sizes. We are going to run train/test n times.
# Each time, we use one fold as the testing test and train a classifier
# from the remaining n-1 folds.
# In this homework, we are going to split based on the indices
# for each instance.

# SplitIndices stores the indices for each train/test run,
# Indices for the training set and the testing set 
# are respectively in two lists named 
# `train` and `test`.

SplitIndices = namedtuple("SplitIndices", ["train", "test"])

def split_cv(length, num_folds):
    """
    This function splits index [0, length - 1) into num_folds (train, test) tuples.
    """
    splits = [SplitIndices([], []) for _ in range(num_folds)]
    indices = list(range(length))
    random.shuffle(indices)
    # Finish this function to populate `splits`.
    # All the indices are split into num_folds splits.
    # Each fold is the testing set in a split, and the remaining indices
    # are added to the corresponding training set.
    fold_size = int(length / num_folds)
    test_start = 0
    for j in range(num_folds):
        for i in range(length):
            if i in range(test_start, test_start + fold_size):
                splits[j].test.append(indices[i])
            else:
                splits[j].train.append(indices[i])
        test_start += fold_size


    return splits


def cv_performance(x, y, num_folds, k):
    """This function evaluates average accuracy in cross validation."""
    length = len(y)
    splits = split_cv(length, num_folds)
    accuracy_array = []

    for split in splits:
        # initialize lists for training and testing data and labels
        # may need to use numPy arrays instead of lists
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        # use indices in split to populate 
        # training and testing data and label lists
        for i in split.train:
            train_data.append(x[i,:])
            train_labels.append(y[i])
        for i in split.test:
            test_data.append(x[i,:])
            test_labels.append(y[i])
        # initialize knn using training data
        knn = Knearest(train_data, train_labels, k)
        # test against test data
        confusion = knn.confusion_matrix(test_data, test_labels)
        # get accuracy
        accuracy = knn.accuracy(confusion)
        
        accuracy_array.append(accuracy)

    return np.mean(accuracy_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()
    
    data = Numbers("../data/mnist.pkl.gz")
    x, y = data.train_x, data.train_y
    if args.limit > 0:
        x, y = x[:args.limit], y[:args.limit]
    best_k, best_accuracy = -1, 0
    for k in [1, 3, 5, 7, 9]:
        accuracy = cv_performance(x, y, 5, k)
        print("%d-nearest neighber accuracy: %f" % (k, accuracy))
        if accuracy > best_accuracy:
            best_accuracy, best_k = accuracy, k
    knn = Knearest(x, y, best_k)
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    accuracy = knn.accuracy(confusion)
    print("Accuracy for chosen best k= %d: %f" % (best_k, accuracy))


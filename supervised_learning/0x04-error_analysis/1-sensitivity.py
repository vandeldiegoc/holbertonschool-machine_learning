#!/usr/bin/env python3
""" module """

import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix:

    Args:
        confusion (numpy.ndarray of shape (classes, classes)):
                    [row indices represent the correct labels
                    and column indices represent the predicted labels]
    """
    tp = np.diagonal(confusion)
    fn = np.sum(confusion, axis=1) - tp
    return(tp / (tp + fn))

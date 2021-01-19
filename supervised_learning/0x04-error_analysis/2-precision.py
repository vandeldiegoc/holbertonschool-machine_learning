#!/usr/bin/env python3
""" module """

import numpy as np


def precision(confusion):
    """calculates the precision for each class in a confusion matrix:

    Args:
        confusion (numpy.ndarray of shape (classes, classes)):
                    [row indices represent the correct labels
                    and column indices represent the predicted labels]
    """
    tp = np.diagonal(confusion)
    fp = np.sum(confusion, axis=0) - tp
    return(tp / (tp + fp))

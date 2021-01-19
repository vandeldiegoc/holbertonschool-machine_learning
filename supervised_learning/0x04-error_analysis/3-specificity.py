#!/usr/bin/env python3
""" module """

import numpy as np


def specificity(confusion):
    """calculates the precision for each class in a confusion matrix:

    Args:
        confusion (numpy.ndarray of shape (classes, classes)):
                    [row indices represent the correct labels
                    and column indices represent the predicted labels]
    """

    tp = np.diagonal(confusion)
    fn = np.sum(confusion, axis=1) - tp
    fp = np.sum(confusion, axis=0) - tp
    tn = np.sum(confusion) - (fp + fn + tp)
    tnr = tn / (fp + tn)
    return(tnr)

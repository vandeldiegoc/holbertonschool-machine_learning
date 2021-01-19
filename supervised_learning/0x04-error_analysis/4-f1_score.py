#!/usr/bin/env python3
""" module """

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the precision for each class in a confusion matrix:

    Args:
        confusion (numpy.ndarray of shape (classes, classes)):
                    [row indices represent the correct labels
                    and column indices represent the predicted labels]
    """
    ppv = precision(confusion)
    tpr = sensitivity(confusion)
    f1 = 2 * ((ppv * tpr) / (ppv + tpr))
    return(f1)

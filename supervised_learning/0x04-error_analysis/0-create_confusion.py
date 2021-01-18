#!/usr/bin/env python3
""" module """

import numpy as np


def create_confusion_matrix(labels, logits):
    """that creates a confusion matrix

    Args:
        labels ([numpy.ndarray of shape (m, classes)]):
                containing the predicted labels
        logits (numpy.ndarray of shape (classes, classes)):
                with row indices representing the correct labels and column
                indices representing the predicted labels
    """
    matrix = np.matmul(labels.T, logits)
    return(matrix)

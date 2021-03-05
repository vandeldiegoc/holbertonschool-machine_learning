#!/usr/bin/env python3
"""module"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ determines if you should stop gradient descent early """
    if opt_cost - cost > threshold:
        return False, 0
    else:
        count += 1
        if count < patience:
            return False, count
        else:
            return True, count

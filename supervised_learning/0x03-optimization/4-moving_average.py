#!/usr/bin/env python3
""" module """


def moving_average(data, beta):
    """hat calculates the weighted
       moving average of a data set:"""
    value = 0
    moving_averages = []
    for i in range(len(data)):
        value = beta * value + (1 - beta) * data[i]
        b_c = value / (1 - beta ** (i + 1))
        moving_averages.append(b_c)
    return moving_averages

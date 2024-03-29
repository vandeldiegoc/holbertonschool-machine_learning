#!/usr/bin/env python3
"""module"""
import sklearn.cluster


def kmeans(X, k):
    """performs K-means on a dataset """
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss

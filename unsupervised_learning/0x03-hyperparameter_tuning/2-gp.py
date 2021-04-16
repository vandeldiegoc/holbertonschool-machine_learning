#!/usr/bin/env python3
"""module  """
import numpy as np


class GaussianProcess:
    """that instantiates a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """constructor"""

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """the covariance kernel matrix"""
        a = np.sum(X1 ** 2, axis=1, keepdims=True)
        b = np.sum(X2 ** 2, axis=1, keepdims=True)
        c = np.matmul(X1, X2.T)
        dist_sq = a + b.reshape(1, -1) - 2 * c
        K = (self.sigma_f ** 2) * np.exp(-0.5 * (1 / (self.l ** 2)) * dist_sq)
        return K

    def predict(self, X_s):
        """ Predicts the mean and standard
            deviation of points in Gaussian process GP"""
        X = self.X
        Y = self.Y
        kernel = self.kernel
        K = kernel(X, X)
        K_s = kernel(X, X_s)
        K_ss = kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)
        mu_s = K_s.T.dot(K_inv).dot(Y).reshape((-1,))
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        return mu_s, np.diag(cov_s)

    def update(self, X_new, Y_new):
        """ Updates a Gaussian Process"""
        self.X = np.append(self.X, X_new).reshape((-1, 1))
        self.Y = np.append(self.Y, Y_new).reshape((-1, 1))
        self.K = self.kernel(self.X, self.X)

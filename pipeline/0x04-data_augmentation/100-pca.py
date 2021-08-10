#!/usr/bin/env python3
"""module"""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """ that performs PCA color augmentation"""
    img_rs = tf.keras.preprocessing.image.img_to_array(image)
    renorm_image = np.reshape(img_rs, (img_rs.shape[0] * img_rs.shape[1], 3))

    mean = np.mean(renorm_image, axis=0)
    img_centered = renorm_image - mean
    std = np.std(img_centered, axis=0)
    img_centered /= std
    cov = np.cov(img_centered, rowvar=False)

    lambdas, p = np.linalg.eig(cov)

    delta = np.dot(p, alphas*lambdas)

    pca_augmentation_version_renorm_image = img_centered + delta
    pca = pca_augmentation_version_renorm_image * std + mean
    pca = np.maximum(np.minimum(pca, 255), 0).astype('uint8')
    pca = pca.reshape((img_rs.shape[0], img_rs.shape[1], 3))
    return pca

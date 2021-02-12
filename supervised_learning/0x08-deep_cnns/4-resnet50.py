#!/usr/bin/env python3
"""module"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """builds the ResNet-50 architecture"""
    X = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                            kernel_initializer="he_normal", padding="same")(X)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation("relu")(norm1)
    max1 = K.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(act1)

    proj1 = projection_block(max1, [64, 64, 256], s=1)
    identi1 = identity_block(proj1, [64, 64, 256])
    identi2 = identity_block(identi1, [64, 64, 256])

    proj2 = projection_block(identi2, [128, 128, 512], s=2)
    identi3 = identity_block(proj2, [128, 128, 512])
    identi4 = identity_block(identi3, [128, 128, 512])
    identi5 = identity_block(identi4, [128, 128, 512])

    proj3 = projection_block(identi5, [256, 256, 1024], s=2)
    identi6 = identity_block(proj3, [256, 256, 1024])
    identi7 = identity_block(identi6, [256, 256, 1024])
    identi8 = identity_block(identi7, [256, 256, 1024])
    identi9 = identity_block(identi8, [256, 256, 1024])
    identi10 = identity_block(identi9, [256, 256, 1024])

    proj4 = projection_block(identi10, [512, 512, 2048], s=2)
    identi11 = identity_block(proj4, [512, 512, 2048])
    identi12 = identity_block(identi11, [512, 512, 2048])

    avg = K.layers.AveragePooling2D(pool_size=7)(identi12)
    output = K.layers.Dense(1000, activation="softmax")(avg)
    model = K.models.Model(inputs=X, outputs=output)
    return(model)

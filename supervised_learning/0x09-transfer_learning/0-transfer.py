#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
import numpy as np


def preprocess_data(X,Y):
    X_p = K.applications.vgg16.preprocess_input(X.astype("float64"))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test) 
    vgg_model = K.applications.VGG16(weights="imagenet", include_top=False, pooling='max')
    vgg_model.trainable = False
    set_trainable = False
    for l in vgg_model.layers:
        if l.name in ['block4_conv1']:
            set_trainable = True
        if set_trainable:
            l.trainable = True

    output = vgg_model.layers[-1].output
    output = K.layers.Flatten()(output)
    vgg = K.Model(vgg_model.input, output)
    model = K.Sequential()
    model.add(K.layers.UpSampling2D())
    model.add(K.layers.BatchNormalization())
    model.add(vgg)
    model.add(K.layers.Dense(512, activation='relu'))
    model.add(K.layers.Dropout(0.3))
    model.add(K.layers.Dense(256, activation='relu'))
    model.add(K.layers.Dropout(0.3))
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dropout(0.3))
    classes = 10
    model.add(K.layers.Dense(classes, activation='softmax'))


    opt = K.optimizers.Adam(lr=1e-5)
    for i in model.layers:
        print(i.name)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['acc'])
    verbo = 1

    history = model.fit(x=x_train, y=y_train,
                        validation_data=(x_test, y_test),
                        epochs=30,
                        batch_size=128,
                        verbose=verbo,
                        steps_per_epoch=100,
                        shuffle=True,
                        validation_steps=15)
    #https://colab.research.google.com/drive/18u5yEPlgSU8jxQXNBGtUtQAIs8coCfRG?usp=sharing
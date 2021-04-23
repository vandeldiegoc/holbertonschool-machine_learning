#!/usr/bin/env python3
"""module"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """that creates a variational autoencoder """
    Input = keras.Input
    UpSampling2D = keras.layers.UpSampling2D
    Model = keras.Model
    Conv2D = keras.layers.Conv2D
    MaxPooling2D = keras.layers.MaxPooling2D

    input_img = Input(input_dims)
    encoded = input_img
    for f in filters:
        encoded = Conv2D(f, (3, 3), activation='relu', padding='same')(encoded)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)

    encoded_img = Input(latent_dims)
    decoded = encoded_img
    for i in reversed(range(1, len(filters))):
        f = filters[i]
        decoded = Conv2D(f, (3, 3), activation='relu', padding='same')(decoded)
        decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(
        filters[0],
        (3,
         3),
        activation='relu',
        padding='valid')(decoded)
    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(input_dims[-1], (3, 3),
                     activation='sigmoid', padding='same')(decoded)

    encoder = Model(input_img, encoded)
    decoder = Model(encoded_img, decoded)
    out_decoder = decoder(encoder(input_img))
    auto = Model(input_img, out_decoder)
    auto.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder, decoder, auto

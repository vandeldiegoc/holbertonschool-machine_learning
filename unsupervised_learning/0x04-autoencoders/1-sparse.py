#!/usr/bin/env python3
""" module """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """ that creates an autoencoder """
    Dense = keras.layers.Dense
    Model = keras.Model
    regularizers = keras.regularizers.l1(lambtha)

    input_img = keras.Input((input_dims,))
    encoded = input_img
    for hidden_layer in hidden_layers:
        encoded = Dense(hidden_layer, activation='relu')(encoded)
    encoded = Dense(latent_dims,
                    activation='relu',
                    activity_regularizer=regularizers)(encoded)

    encoded_img = keras.Input((latent_dims,))
    decoded = encoded_img
    for hidden_layer in reversed(hidden_layers):
        decoded = Dense(hidden_layer, activation='relu')(decoded)
    decoded = Dense(input_dims, activation="sigmoid")(decoded)

    encoder = Model(input_img, encoded)
    decoder = Model(encoded_img, decoded)
    out_decoder = decoder(encoder(input_img))
    auto = Model(input_img, out_decoder)
    auto.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder, decoder, auto

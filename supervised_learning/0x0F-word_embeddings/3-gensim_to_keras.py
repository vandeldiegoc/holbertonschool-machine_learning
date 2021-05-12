#!/usr/bin/env python3
"""module"""
from tensorflow import keras


def gensim_to_keras(model):
    """ converts a gensim word2vec
        model to a keras Embedding layer: """
    layer_keras = model.wv.get_keras_embedding(train_embeddings=True)
    return layer_keras

#!/usr/bin/env python3
"""module"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """ that creates a bag of words embedding matrix:"""
    vectorizer = CountVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    embedding = x.toarray()
    features = vectorizer.get_feature_names()

    return embedding, features

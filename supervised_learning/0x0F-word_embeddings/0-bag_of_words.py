#!/usr/bin/env python3
"""module"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """ that creates a bag of words embedding matrix:"""
    vectorizer = CountVectorizer()
    if vocab is None:
        X = vectorizer.fit_transform(sentences)
    else:
        X = vectorizer.fit_transform(sentences)
    e = X.toarray()
    return e, vectorizer.get_feature_names()

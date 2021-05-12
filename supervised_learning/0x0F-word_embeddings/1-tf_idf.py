#!/usr/bin/env python3
"""module"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ that creates a bag of words embedding matrix:"""
    vectorizer = TfidfVectorizer()
    if vocab is None:
        X = vectorizer.fit_transform(sentences)
    else:
        X = vectorizer.fit_transform(vocab)
    e = X.toarray()
    f = vectorizer.get_feature_names()
    return e, f

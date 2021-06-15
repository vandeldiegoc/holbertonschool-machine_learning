#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


def semantic_search(corpus_path, sentence):
    """that performs semantic search on a corpus of document"""
    text_file = [sentence]
    for file in os.listdir(corpus_path):
        if file.endswith(".md"):
            with open(corpus_path + "/" + file, 'r') as f:
                text = f.read()
                text_file.append(text)
    emb = embed(text_file)
    correlation = np.inner(emb, emb)
    argm = np.argmax(correlation[0, 1:]) + 1
    return text_file[argm]

#!/usr/bin/env python3
""" module """
import tensorflow_datasets as tfds
import numpy


class Dataset:
    """load en-pt lengues dataset and tokenize sentece"""
    def __init__(self):
        """costructor """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

    def tokenize_dataset(self, data):
        """ tokenize for return 2 tensor """
        # tokenizer_en = \
        #     tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        #         (en.numpy() for pt, en in data),
        #         target_vocab_size=2**15)
        tokenizer_en = \
            tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for pt, en in data),
                target_vocab_size=2**15)
        # vtokenizer_pt = \
        #     tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        #         (pt.numpy() for pt, en in data),
        #         target_vocab_size=2**15)
        tokenizer_pt = \
            tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, en in data),
                target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """  that encodes a translation into tokens"""
        pt = [self.tokenizer_pt.vocab_size] + \
            self.tokenizer_pt.encode(pt.numpy()) + \
            [self.tokenizer_pt.vocab_size+1]
        en = [self.tokenizer_en.vocab_size] + \
            self.tokenizer_en.encode(en.numpy()) + \
            [self.tokenizer_en.vocab_size+1]
        return pt, en

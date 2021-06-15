#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
question_answer = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def qa_bot(coprus_path):
    """that answers questions from multiple reference texts"""
    command = ['exit', 'quit', 'goodbye', 'bye']
    while(True):
        arg = input("Q: ")
        if arg.strip() in command:
            print("A: Goodbye")
            break
        text = semantic_search(coprus_path, arg)
        answer = question_answer(arg, text)
        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))

#!/usr/bin/env python3
""" module """
import pymongo


def insert_school(mongo_collection, **kwargs):
    """ inserts a new document in a collection based on kwargs:"""
    _id = mongo_collection.insert_one(kwargs).inserted_id
    return (_id)

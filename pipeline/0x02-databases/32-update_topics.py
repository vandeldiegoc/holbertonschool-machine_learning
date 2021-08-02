#!/usr/bin/env python3
""" module """
import pymongo


def update_topics(mongo_collection, name, topics):
    """returns the list of school having a specific topic"""
    mongo_collection.update_many({"name": name}, {"$set": {"topics": topics}})

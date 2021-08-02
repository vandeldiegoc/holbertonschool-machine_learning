#!/usr/bin/env python3
"""module"""


def schools_by_topic(mongo_collection, topic):
    """returns the list of school having a specific topic"""
    res = []
    returned_values = mongo_collection.find({"topics": {"$all": [topic]}})
    for value in returned_values:
        res.append(value)
    return res

#!/usr/bin/env python3
"""module"""

from pymongo import MongoClient

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    school_collection = client.logs.nginx
    method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print('{} logs'.format(school_collection.count_documents({})))
    print('Methods:')
    for meth in method:
        print('\tmethod {}: {}'.format(
            meth, school_collection.count_documents({'method': meth})))
    print('{} status check'.format(school_collection.count_documents(
        {'method': 'GET', 'path': '/status'})))

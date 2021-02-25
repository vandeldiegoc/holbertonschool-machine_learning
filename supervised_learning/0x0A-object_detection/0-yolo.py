#!/usr/bin/env python3
"""module"""
import tensorflow.keras as K


class  Yolo:
    """class Yolo that uses the Yolo v3
       algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            print(f)
            self.class_names = f.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

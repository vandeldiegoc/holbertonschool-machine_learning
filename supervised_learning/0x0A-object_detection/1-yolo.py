#!/usr/bin/env python3
"""module"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """class Yolo that uses the Yolo v3
       algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """sigmoid_activation"""
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """ Write a class Yolo (Based on 0-yolo.py): """
        img_h, img_w = image_size[0], image_size[1]
        for i in range(len(outputs)):
            boxes, box_confidences, box_class_probs = [], [], []
            input_w = self.model.input_shape[1]
            input_h = self.model.input_shape[2]

            grid_h = outputs[i].shape[0]
            grid_w = outputs[i].shape[1]
            anchor_boxes = outputs[i].shape[2]

            tx = outputs[i][..., 0]
            ty = outputs[i][..., 1]
            tw = outputs[i][..., 2]
            th = outputs[i][..., 3]

            c = np.zeros((grid_h, grid_w, anchor_boxes))

            idx_y = np.arange(grid_h)
            idx_y = idx_y.reshape(grid_h, 1, 1)
            idx_x = np.arange(grid_w)
            idx_x = idx_x.reshape(1, grid_w, 1)
            cx = c + idx_x
            cy = c + idx_y

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bx = self.sigmoid(tx) + cx
            by = self.sigmoid(ty) + cy
            bw = pw * np.exp(tw)
            bh = ph * np.exp(th)

            bx = bx / grid_w
            by = by / grid_h

            bw = bw / input_w
            bh = bh / input_h

            bx1 = bx - bw / 2
            by1 = by - bh / 2
            bx2 = bx + bw / 2
            by2 = by + bh / 2

            outputs[i][..., 0] = bx1 * img_w
            outputs[i][..., 1] = by1 * img_h
            outputs[i][..., 2] = bx2 * img_w
            outputs[i][..., 3] = by2 * img_h

            boxes.append(outputs[i][..., 0:4])
            box_confidences.append(self.sigmoid(outputs[i][..., 4:5]))
            box_class_probs.append(self.sigmoid(outputs[i][..., 5:]))
        return(boxes, box_confidences, box_class_probs)

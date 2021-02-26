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
        image_height, image_width = image_size[0], image_size[1]

        boxes = [output[..., 0:4] for output in outputs]

        for i, box in enumerate(boxes):
            grid_height, grid_width, anchor_boxes, _ = box.shape

            # BONDING BOX CENTER COORDINATES (x,y)
            c = np.zeros((grid_height, grid_width, anchor_boxes), dtype=int)

            # height
            indexes_y = np.arange(grid_height)
            indexes_y = indexes_y.reshape(grid_height, 1, 1)
            cy = c + indexes_y

            # width
            indexes_x = np.arange(grid_width)
            indexes_x = indexes_x.reshape(1, grid_width, 1)
            cx = c + indexes_x

            # darknet center coordinates output
            tx = (box[..., 0])
            ty = (box[..., 1])

            # normalized output
            tx_n = self.sigmoid(tx)
            ty_n = self.sigmoid(ty)

            # placement within grid
            bx = tx_n + cx
            by = ty_n + cy

            # normalize to grid
            bx /= grid_width
            by /= grid_height

            # BONDING BOX WIDTH AND HEIGHT (w, h)
            # darknet output
            tw = (box[..., 2])
            th = (box[..., 3])

            # log-space transformation
            tw_t = np.exp(tw)
            th_t = np.exp(th)

            # anchors box dimensions [anchor_box_width, anchor_box_height]
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # scale to anchors box dimensions
            bw = pw * tw_t
            bh = ph * th_t

            # normalizing to model input size
            input_width = self.model.input.shape[1].value
            input_height = self.model.input.shape[2].value
            bw /= input_width
            bh /= input_height

            # BOUNDING BOX CORNER COORDINATES
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh

            # scaling to image size
            box[..., 0] = x1 * image_width
            box[..., 1] = y1 * image_height
            box[..., 2] = x2 * image_width
            box[..., 3] = y2 * image_height

        box_confidences = \
            [self.sigmoid(output[..., 4, np.newaxis]) for output in outputs]
        box_class_probs = \
            [self.sigmoid(output[..., 5:]) for output in outputs]

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """return filtered boxe"""
        scores = []

        for i in range(len(boxes)):
            scores.append(box_confidences[i] * box_class_probs[i])

        filter_boxes = [box.reshape(-1, 4) for box in boxes]
        filter_boxes = np.concatenate(filter_boxes)

        classes = [np.argmax(box, -1) for box in scores]
        classes = [box.reshape(-1) for box in classes]
        classes = np.concatenate(classes)

        class_scores = [np.max(box, -1) for box in scores]
        class_scores = [box.reshape(-1) for box in class_scores]
        class_scores = np.concatenate(class_scores)

        filtering_mask = np.where(class_scores >= self.class_t)
        filtered_boxes = filter_boxes[filtering_mask]
        box_classes = classes[filtering_mask]
        box_scores = class_scores[filtering_mask]

        return(filtered_boxes, box_classes, box_scores)

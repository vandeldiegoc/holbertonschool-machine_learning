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

    def iou(self, filtered_boxes, scores):
        """Returns the intersection over union result"""
        # grab the coordinates of the bounding boxes
        x1 = filtered_boxes[:, 0]
        y1 = filtered_boxes[:, 1]
        x2 = filtered_boxes[:, 2]
        y2 = filtered_boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = scores.argsort()[::-1]

        # initialize the list of picked indexes
        pick = []
        # keep looping while some indexes still remain in the indexes
        # list
        while idxs.size > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            i = idxs[0]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            inter = w * h
            union = area[i] + area[idxs[1:]] - inter
            overlap = inter / union

            # delete all indexes from the index list that have
            ind = np.where(overlap <= self.nms_t)[0]
            idxs = idxs[ind + 1]

        # return only the bounding boxes that were picked using the
        # integer data type
        return pick

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """non max suppression"""
        box_predictions = []
        predicted_box_classes, predicted_box_score = [], []
        u_classes = np.unique(box_classes)
        for cls in u_classes:
            idx = np.where(box_classes == cls)

            filters = filtered_boxes[idx]
            scores = box_scores[idx]
            classes = box_classes[idx]

            pick = self.iou(filters, scores)

            filters1 = filters[pick]
            scores1 = scores[pick]
            classes1 = classes[pick]

            box_predictions.append(filters1)
            predicted_box_classes.append(classes1)
            predicted_box_score.append(scores1)
        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_score = np.concatenate(predicted_box_score, axis=0)

        return (box_predictions, predicted_box_classes, predicted_box_score)

import os
import PIL
import scipy.io
import argparse
import scipy.misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from DeepLearning.CourseFour.yolo_utils import *
from DeepLearning.CourseFour.yad2k.models.keras_yolo import *


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    """
    Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    """
    box_scores = box_confidence * box_class_probs

    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)

    filtering_mask = box_class_scores >= threshold

    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)

    return scores, boxes, classes


def iou(box1, box2):
    """
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)

    Returns:
    iou -- intersection over union (IoU) between box1 and box2
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (yi2 - yi1) * (xi2 - xi1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=.5):
    """
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    """
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=0.5):
    """
   Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along
   with their scores, box coordinates and classes.

   Arguments:
   yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                   box_confidence: tensor of shape (None, 19, 19, 5, 1)
                   box_xy: tensor of shape (None, 19, 19, 5, 2)
                   box_wh: tensor of shape (None, 19, 19, 5, 2)
                   box_class_probs: tensor of shape (None, 19, 19, 5, 80)
   image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
   max_boxes -- integer, maximum number of predicted boxes you'd like
   score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
   iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

   Returns:
   scores -- tensor of shape (None, ), predicted score for each box
   boxes -- tensor of shape (None, 4), predicted box coordinates
   classes -- tensor of shape (None,), predicted class for each box
   """
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs[:]

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    """
    image, image_data = preprocess_image("./images/" + image_file, model_image_size=(608, 608))
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save(os.path.join("out", image_file), quality=90)
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    return out_scores, out_boxes, out_classes



sess = K.get_session()
class_names = read_classes("./model_data/coco_classes.txt")
anchors = read_anchors("./model_data/yolo_anchors.txt")
image_shape = (720., 1280.)
yolo_model = load_model("./model_data/yolo.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)


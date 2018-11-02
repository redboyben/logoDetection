# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 22:51:11 2018

@author: Benoit
"""

#runfile('D:/VideoObjectDetection/detection_functions.py', wdir='D:/VideoObjectDetection')

# Todo: import only what is required
import numpy as np
import sys
import tensorflow as tf
import cv2

sys.path.append("models/research/")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



def videoReader(vidCap):
  modelName = 'sysnav_logo_inference_graph3'
  pathToCheckpoint = modelName + '/frozen_inference_graph.pb'
  #opener = urllib.request.URLopener()
  #opener.retrieve(url_base + modelFile, modelFile) # url_base not defined

  print("Load trained model")
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(pathToCheckpoint, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  category_index = {1: {'id': 1, 'name': 'logo'}} # We have only one category here
  
  print("Begining treatment of the video")
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        
      while True:
        ret, image_np = vidCap.read()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # Actual detection.
        (boxes, scores, classes) = sess.run(
            [boxes, scores, classes],
            feed_dict={image_tensor: image_np_expanded})
        # Applying threshold
        boxes, scores = keepHighScoreBoxes(boxes, scores)
        
        boxes = np.array([convertToCoordinates(box, image_np.shape[0:2]) for box in boxes])
        # Going through the boxes and discriminating the false positive as much as possible
        
        relevantBoxesIndex = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            image_box = image_np[box[0]:box[2],box[1]:box[3],:]
            if(isValidBox(image_box)):
                relevantBoxesIndex.append(i)
                
                
                
        boxes = boxes[relevantBoxesIndex]
        
        
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            boxes,
            np.squeeze(classes).astype(np.int32),
            scores,
            category_index,
            use_normalized_coordinates=False,
            line_thickness=4)
                        
            
        cv2.imshow('object detection', cv2.resize(image_np, (640,480)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
def keepHighScoreBoxes(boxes, scores, threshold = .5):
    scores = scores[scores>threshold] # the list is already sorted
    num_predictions = len(scores)
    return boxes[0][0:num_predictions], scores

def isValidBox(image_box):
    return(colorValid(image_box) and shapeValid(image_box))

def shapeValid(image_box, threshold = 2):
    ratio = float(image_box.shape[0])/float(image_box.shape[1])
    return(ratio < threshold and 1/ratio < threshold)

def colorValid(image_box):
    box_air = image_box.shape[0]*image_box.shape[1]
    new_image = image_box.copy()
    isRedCount = 0
    for i in range(image_box.shape[0]):
        for j in range(image_box.shape[1]):
            b, g, r = image_box[i][j]
            isRed = (r> 40 and r>max(1.5*b, 1.5*g))
            if(isRed):
                new_image[i][j] = [255, 0, 0]
                isRedCount += 1
            else:
                new_image[i][j] = [0, 0, 0]
    ratio = round(float(isRedCount)/float(box_air), 3)
    return(ratio >= 0.025 and ratio <= 0.25)


def convertToCoordinates(box, im_dim):
    im_height, im_width = im_dim
    ymin = min(int(box[0] * im_height), im_height)
    xmin = min(int(box[1] * im_width), im_width)
    ymax = min(int(box[2] * im_height), im_height)
    xmax = min(int(box[3] * im_width), im_width)
    return(np.array([ymin, xmin, ymax, xmax]))

v = cv2.VideoCapture('sysnav/positive.avi')
tmp = videoReader(v)

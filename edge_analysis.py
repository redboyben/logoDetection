# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:26:55 2018

@author: Benoit
"""

import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt




def keepRelevantBoxes(boxes, scores, threshold = .5):
    scores = scores[scores>threshold] # the list is already sorted
    num_predictions = len(scores)
    return boxes[0][0:num_predictions], scores


def convertToCoordinates(box, im_dim):
    im_height, im_width = im_dim
    ymin = min(int(box[0] * im_height), im_height)
    xmin = min(int(box[1] * im_width), im_width)
    ymax = min(int(box[2] * im_height), im_height)
    xmax = min(int(box[3] * im_width), im_width)
    return(np.array([ymin, xmin, ymax, xmax]))



sys.path.append("models/research/")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
modelName = 'sysnav_logo_inference_graph3'
pathToCheckpoint = modelName + '/frozen_inference_graph.pb'
pathToModelLabels = os.path.join('trainSysnav', 'object-detection.pbtxt')
maxNumberOfClass = 1
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

    print("Loading labels")
    label_map = label_map_util.load_labelmap(pathToModelLabels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=maxNumberOfClass, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    print("Begining treatment of the video")
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
        
    
            for image_name in os.listdir('sysnav/imagesTest/'):
#      while True:
#        ret, image_np = vidCap.read()
                try:
                    image_np = cv2.imread('sysnav/imagesTest/' + image_name)
                except:
                    continue
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
                boxes, scores = keepRelevantBoxes(boxes, scores)
                
                boxes = np.array([convertToCoordinates(box, image_np.shape[0:2]) for box in boxes])
                
                # Visualization of the results of a detection.
                stop = False
                minT = 30
                maxT = 150
                for box in boxes:
                    img = image_np[box[0]:box[2],box[1]:box[3],:]
                #            red_distrib = image_np[box[0]:box[2],box[1]:box[3],0]
                #            plt.hist(red_distrib, bins=10)
                #            plt.show()
                #        
                    
                #            showCircles(image_box)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    '''
                     * Function to perform Canny edge detection and display the
                     * result. 
                    '''
                    def cannyEdge():
                        global img, minT, maxT
                        print(minT)
                        edge = cv2.Canny(img, minT, maxT)
                        cv2.imshow("edges", edge)
                    
                    '''
                     * Callback function for minimum threshold trackbar.
                    ''' 
                    def adjustMinT(v):
                    	global minT
                    	minT = v
                    	cannyEdge()
                    
                    '''
                     * Callback function for maximum threshold trackbar.
                    '''
                    def adjustMaxT(v):
                    	global maxT
                    	maxT = v
                    	cannyEdge()
                    	
                    
                    # set up display window with trackbars for minimum and maximum threshold
                    # values
                    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
                    cv2.createTrackbar("minT", "edges", minT, 255, adjustMinT)
                    cv2.createTrackbar("maxT", "edges", maxT, 255, adjustMaxT)
                    
                    # perform Canny edge detection and display result
                    cannyEdge()
                    cv2.waitKey(0)
    
            
            
                      
                if(stop):
                    break

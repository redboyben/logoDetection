#runfile('D:/VideoObjectDetection/detection_functions.py', wdir='D:/VideoObjectDetection')

# Todo: import only what is required
import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import time
import matplotlib.pyplot as plt
from scipy.misc import imsave
from PIL import Image

sys.path.append("models/research/")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



def videoReader(vidCap):
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
        
      base_url = 'sysnav/imagesTest/'
#      for image_name in os.listdir(base_url):
      while True:
        ret, image_np = vidCap.read()
#        if(image_name != "scene00701.png"):
#            continue
#        try:
#            image_np = cv2.imread(base_url + image_name)
#            if(image_name[-11:] == "distrib.png" or image_name[-8:] == "zoom.png"):
#                raise Exception("Don't care of the hists")
#        except:
#            continue
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
        
#        stop = False
#        box_count = 0
#        boxes_base_url = "sysnav/boxes/"
        relevantBoxesIndex = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
#            box_count += 1
            image_box = image_np[box[0]:box[2],box[1]:box[3],:]
#            imsave(boxes_base_url + image_name[:-4] + "_box_"  + str(box_count) + "_zoom.png", image_box)
            
#            saveHist(image_box,boxes_base_url + image_name, box_count)
            
#            showCircles(image_box)
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
#            while(not (cv2.waitKey(25) & 0xFF == ord('n'))):
#                if cv2.waitKey(25) & 0xFF == ord('q'):
#                  cv2.destroyAllWindows()
#                  stop = True
#                  break
#                pass
#        if(stop):
#            break
#             to remove later
#            cv2.destroyAllWindows()
        
def keepHighScoreBoxes(boxes, scores, threshold = .5):
    scores = scores[scores>threshold] # the list is already sorted
    num_predictions = len(scores)
    return boxes[0][0:num_predictions], scores

def isValidBox(image_box):
    return(colorValid(image_box) and shapeValid(image_box))

def shapeValid(image_box, threshold = 2):
    ratio = float(image_box.shape[0])/float(image_box.shape[1])
#    if(ratio >= threshold):
#        print("Box too tall", ratio)
#    if(1/ratio >= threshold):
#        print("Box too large", ratio)
    return(ratio < threshold and 1/ratio < threshold)

#def colorValid(image_box, bu_and_image_name, box_count):
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
#    plt.imshow(new_image, interpolation='nearest')
    
#    plt.imshow(image_box, interpolation='nearest')
#    plt.show()    
#    print("final decision")
    ratio = round(float(isRedCount)/float(box_air), 3)
#    if(ratio < 0.004):
#        print("Not enough red", ratio)
#        return(False)
#    if(ratio > 0.25):
#        print("too much red", ratio)
#        return(False)
    return(ratio >= 0.025 and ratio <= 0.25)
#    print(ratio)
#    plt.savefig(bu_and_image_name[:-4] + "_" + str(box_count) + "_red_"+ str(ratio) + "_detect_zoom.png")



def convertToCoordinates(box, im_dim):
    im_height, im_width = im_dim
    ymin = min(int(box[0] * im_height), im_height)
    xmin = min(int(box[1] * im_width), im_width)
    ymax = min(int(box[2] * im_height), im_height)
    xmax = min(int(box[3] * im_width), im_width)
    return(np.array([ymin, xmin, ymax, xmax]))

def saveHist(image_box, bu_and_image_name, box_count):
    
    red_distrib = image_box[:,:,0].flatten()
    plt.hist(red_distrib, bins=10)
    
    blue_distrib = image_box[:,:,1].flatten()
    plt.hist(blue_distrib, bins=10)
    
    green_distrib = image_box[:,:,2].flatten()
    plt.hist(green_distrib, bins=10)
    plt.savefig(bu_and_image_name[:-4] + "_" + str(box_count) + "_color_distrib.png")
    plt.close()


def showCircles(image_box):
    gray = cv2.cvtColor(image_box, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, .5, rows / 3,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(image_box, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(image_box, center, radius, (255, 0, 255), 3)


v = cv2.VideoCapture('sysnav/positive.avi')
#v = cv2.VideoCapture(0)
tmp = videoReader(v)

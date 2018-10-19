#runfile('D:/VideoObjectDetection/detection_functions.py', wdir='D:/VideoObjectDetection')

# Todo: import only what is required
import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import time

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
        boxes = keepRelevantBoxes(boxes, scores)
        boxes = np.array([convertToCoordinates(box, image_np.shape[0:2]) for box in boxes])
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            boxes,
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
#            use_normalized_coordinates=True,
            use_normalized_coordinates=False,
            line_thickness=4)

        cv2.imshow('object detection', cv2.resize(image_np, (1100,600)))
        time.sleep(1)

        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break
      
        # to remove later
#        cv2.destroyAllWindows()
        
def keepRelevantBoxes(boxes, scores, threshold = .5):
    scores = scores[scores>threshold] # the list is already sorted
    num_predictions = len(scores)
    return boxes[0][0:num_predictions]


def convertToCoordinates(box, im_dim):
    im_height, im_width = im_dim
    xmin = int(box[0] * im_height)
    xmax = int(box[1] * im_width)
    ymin = int(box[2] * im_height)
    ymax = int(box[3] * im_width)
    return(np.array([xmin, xmax, ymin, ymax]))


def convertToCoordinatesTmp(box, im_dim):
    im_height, im_width = im_dim
    xmax = int(box[0] * im_width)
    xmin = int(box[1] * im_width)
    ymax = int(box[2] * im_height)
    ymin = int(box[3] * im_height)
    return(np.array([[xmin, ymin], [xmax, ymax], [xmax, ymin], [xmin, ymax]]))





v = cv2.VideoCapture('sysnav/positive.avi')
videoReader(v)


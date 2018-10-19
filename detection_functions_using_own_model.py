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
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        print(boxes)
        print(scores)
        print(num_detections)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4)

        cv2.imshow('object detection', cv2.resize(image_np, (1100,600)))
        time.sleep(.3)
        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break

v = cv2.VideoCapture('sysnav/positive.avi')
videoReader(v)

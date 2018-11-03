# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 16:54:55 2018

@author: Benoit
"""

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity("FATAL")
import cv2
import time

def treatVideo(input_file, verbose, display_frequence):
    """Treats the entire video and returns the list of found logos

    Parameters
    ----------
    input_file : str
        Path to the input file
    verbose: boolean
        Displaying evolution in console or not
    display_frequence: integer
        Frequence of display (seconds); not used if verbose is set to False

    Returns
    -------
    array, shape: [N, 5]
        An array containing the detected logos and their frame ids
    """

    # Loading model
    detection_graph = loadDetectionModel()

    print("Treating video")
    # Reading video
    vid_cap = cv2.VideoCapture(input_file)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            frame_id = 1 # To identify the frame and store the results
            result_list = [] # List modified inplace with the results
            ts = time.time() # For time analysis purposes
            last_frame_id_treated = 1 # For time analysis purposes (fps computation)
            while(True): # Broken when ret is false (end of video or could not read)
                # Reading the image
                ret, image = vid_cap.read()
                
                # If end of video, release and stop
                if(not ret):
                    vid_cap.release()
                    if(frame_id == 1): # Meaning the video could not be read
                        raise Exception("The video cannot be read.")
                    break
                
                # Treating the frame; result_list is modified inplace
                treatImage(image, detection_graph, result_list, sess, frame_id)
                
                if(frame_id == 5):
                    return(result_list)

                # Displaying evolution if required by user
                if(verbose):
                    if(time.time() - ts > display_frequence):
                        new_ts = time.time()
                        time_spent = new_ts - ts
                        ts = new_ts
                        printEvolution(frame_id, last_frame_id_treated, time_spent)
                        last_frame_id_treated = frame_id
            
                frame_id += 1
                
    return(result_list)
    
def loadDetectionModel():
    """Load the DL trained model

    Parameters
    ----------
    None

    Returns
    -------
    Graph object (tensorflow)
        The frozen model
    Raises exception if file corrupted or not present 
    """
    print("Loading the frozen model")
    pathToCheckpoint = 'sysnav_logo_inference_graph/frozen_inference_graph.pb'
    # Creating empty graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pathToCheckpoint, 'rb') as fid:
            try:
                serialized_graph = fid.read()
            except:
                raise Exception("Corrupted model. Please use the correct file and location.")
            # Parsing frozen graph
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return(detection_graph)


def printEvolution(frame_id, last_frame_id_treated, time_spent):
    """Print evolution in console

    Parameters
    ----------
    frame_id: integer
        Frame id
    last_frame_id_treated: integer
        Last frame id that was taken into account for last display of evolution
    ts: float
        Current timestamp

    Returns
    -------
    None
        Display only
    """
    secTreated = frame_id / 30 # The video is 30 fps
    fps = round((frame_id - last_frame_id_treated)/time_spent, 1)
    print(str(round(secTreated, 1)) + 
          "s of the video have been treated (" +
          str(int(secTreated*30)) +
          " frames), batch's average fps:" +
          str(fps))

def treatImage(image, detection_graph, result_list, tf_sess, frame_id):
    """Find the boxes of with logos in a frame

    Parameters
    ----------
    image : array, shape: [w, h, 3]
        The frame treated
    detection_graph : Graph object (tensorflow)
        The model already trained
    result_list: list
        List of the results, modified inplace
    tf_sess : Session object (tensorflow)
        The on going tensorflow session
    frame_id: integer
        Frame id

    Returns
    -------
    None
        Modifications are done inplace
    """
    # Retrieving the boxes
    boxes = getBoxes(image, detection_graph, tf_sess)
    
    # Going through the boxes and discriminating the false positive as much as possible
    validateBoxes(boxes, image, result_list, frame_id) # inplace modifications
            
def validateBoxes(boxes, image, result_list, frame_id):
    """Validate the boxes found by the model with independent filters

    Parameters
    ----------
    boxes : array, shape: [N, 4]
        The boxes computed by the DL model
    image : array, shape: [w, h, 3]
        The frame treated
    result_list: list
        List of the results, modified inplace

    Returns
    -------
    None
        Modifications are done inplace
    """
    for i in range(boxes.shape[0]):
        box = boxes[i]
        # Isolating the box detected
        image_box = image[box[0]:box[2],box[1]:box[3],:]
        # Validating or not the box
        if(isValidBox(image_box)):
            addBoxToResult(result_list, frame_id, box)
    
    
def getBoxes(image, detection_graph, tf_sess):
    """Find the boxes of interest with the DL model

    Parameters
    ----------
    image : array, shape: [w, h, 3]
        The frame treated
    detection_graph : Graph object (tensorflow)
        The model already trained
    tf_sess : Session object (tensorflow)
        The on going tensorflow session

    Returns
    -------
    array, shape: [M, 4]
        an array containing the relevant boxes
    """
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # Actual detection.
    (boxes, scores, classes) = tf_sess.run(
            [boxes, scores, classes],
            feed_dict={image_tensor: image_np_expanded})
    # Applying threshold on scores
    boxes, scores = keepHighScoreBoxes(boxes, scores)

    # Convert normalized coordinates to pixels
    boxes = np.array([convertToCoordinates(box, image.shape[0:2]) for box in boxes])
    return(boxes)

def keepHighScoreBoxes(boxes, scores, threshold = .5):
    """Filters the boxes from the DL model with a score under the threshold

    Parameters
    ----------
    boxes : array, shape: [N, 4]
        The boxes computed by the DL model
    scores : list, length: N
        A list of the scores (same order as the boxes by default)
    threshold: integer
        Decision threshold; since it is a probability here, default value is .5

    Returns
    -------
    array, shape: [M, 4]
        an array containing the relevant boxes
    list, length: M
        list of the scores (in case of display)
    """
    scores = scores[scores>threshold] # the list is already sorted
    num_predictions = len(scores)
    return boxes[0][0:num_predictions], scores

def isValidBox(image_box):
    """Applying filters on the cut images given as argument

    Parameters
    ----------
    image_box : array, shape: [I, J]
        A part of an image

    Returns
    -------
    boolean
        The verdict on the status of that box
    """
    return(colorFilter(image_box) and shapeFilter(image_box))

def shapeFilter(image_box, threshold = 2):
    """Filters the boxes from the DL model with a score under the threshold

    Parameters
    ----------
    image_box : array, shape: [I, J]
        A part of an image
    threshold:
        Tolerated maximum ratio between width and height

    Returns
    -------
    boolean:
        Decision regarding the shape of the box
    """
    # max(ratio, inverse of ratio) of image must not be more than the threshold
    ratio = float(image_box.shape[0])/float(image_box.shape[1])
    return(ratio < threshold and 1/ratio < threshold)

def colorFilter(image_box):
    """Filters the boxes from the DL model with a score under the threshold

    Parameters
    ----------
    image_box : array, shape: [I, J]
        A part of an image

    Returns
    -------
    boolean:
        Decision regarding the amount of red in the box
    """
    box_air = image_box.shape[0]*image_box.shape[1]
    redCount = 0
    # Counting the red pixels
    for i in range(image_box.shape[0]):
        for j in range(image_box.shape[1]):
            if(isRed(image_box[i][j])):
                redCount += 1
    # Computing ratio
    ratio = round(float(redCount)/float(box_air), 3)
    return(ratio >= 0.025 and ratio <= 0.25)

def isRed(colors):
    """Decision weither a pixel is red or not

    Parameters
    ----------
    colors : array, length: 3
        The r, g, b values of a pixel

    Returns
    -------
    boolean:
        Decision on weither the pixel is red of not
    """
    b, g, r = colors # OpenCV puts the colors in that order
    return(r > 40 and r > max(1.5*b, 1.5*g))


def convertToCoordinates(box, im_dim):
    """Conversion from normalized coordinates to pixel values

    Parameters
    ----------
    box : array, shape: [4]
        A box delimitation
    im_dim: list, length: 2

    Returns
    -------
    array, shape:[4]
        Box with pixel values
    """
    im_height, im_width = im_dim
    ymin = min(int(box[0] * im_height), im_height)
    xmin = min(int(box[1] * im_width), im_width)
    ymax = min(int(box[2] * im_height), im_height)
    xmax = min(int(box[3] * im_width), im_width)
    # This format is the one used by the vizualisation library used.
    return(np.array([ymin, xmin, ymax, xmax]))

def addBoxToResult(res_list, frame_id, box):
    """Adding validated box to the results

    Parameters
    ----------
    res_list : list, length: N
        The already validated boxes
    frame_id: integer
        Frame id
    box: array, shape:[4]
        The pixel values of the box

    Returns
    -------
    None
        Modifications are done inplace
    """
    # Box values' order is defined in the convertToCoordinates function
    x, y, w, h = box[1], box[0], box[3] - box[1], box[2] - box[0]
    res_list.append(np.array([frame_id, x, y, w, h])) # inplace in python

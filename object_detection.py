import numpy as np
import os
import tensorflow as tf
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util
import pafy
from google.protobuf.message import DecodeError
import show_video


def tfWarnings():
    """
    Make Tensorflow less verbose
    tensorflow will give warning as it suggested using GPU
    below code able to remove the tensorflow warning  
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass


def loadYoutube(url):
    # youtube video
    # url : youtube link
    video=pafy.new(url)
    best=video.getbest(preftype="mp4")

    # youtube video input
    cap = cv2.VideoCapture(1)
    cap.open(best.url)
    return cap


def loadWebcam():
    # use webcam to get video data 
    cap = cv2.VideoCapture(0)
    return cap

def loadMP4(videoName)
    #mp4 video in Video directory
    video =videoName
    try:
        video_dir=os.path.join(CWD_PATH,Model)
    except FileNotFoundError:
        print("Wrong file or file path")

    PATH_TO_VIDEO = os.path.join(video_dir,video)
    cap = cv2.VideoCapture(PATH_TO_VIDEO)
    return cap


def loadURL(url):
    # url : online video url
    cap = cv2.VideoCapture(url)
    return cap

def loadIPCam(ipAddr)
    #ipcam use for input from mobile phone camera 
    #ipAddr : ip address of the ip cam (e.g.http://118.138.7.198:8080/video)
    cap = cv2.VideoCapture(ipAddr)
    return cap

def loadModel()
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def start_detection(cap):
    detection_graph=loadModel() #load the trained model 

    # Define input and output tensors (i.e. data) for the object detection classifier
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            while True:
                # Read frame from camera
                ret, image_np = cap.read()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Extract image tensor (image)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Extract detection boxes
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Extract detection scores
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                # Extract detection classes
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                # Extract number of detections
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                createWindow(image_np,boxes,scores,classes)

                ################################################################################################
                # use theading to create a seperate thread to process the video data windows
                #  this can help to mitigate the lag on video data processing 
                # video_window=show_video.VideoShow(cv2.resize(image_np, (1200, 800))) #create a seperate window for video window
                # video_window.start() #create a seperate thead for the video window
                # video_window.show()  #show the video in seperate window

def createWindow(image_np,boxes,scores,classes):
     # Visualization of the results of a detection.
    # This is where we draw the box around the object
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes), # squeeze() remove single-dimensional entries from the shape of an array.
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.4
        )
    
    # create a new windows, display output frame with the drawn box
    cv2.imshow('PPE Detection', cv2.resize(image_np, (1200, 800)))

    if cv2.waitKey(25) & 0xFF == ord('q'): # quit the window by pressing "q" on keyboard
        # Clean up memory 
        cap.release()
        cv2.destroyAllWindows()
        break


def main():
    # remove tensorflow warning
    tfWarnings() 

    #current working directory
    CWD_PATH = os.getcwd()

    # path to the frozen graph:
    
    frozenModel='frozen_inference_graphX.pb'
    try:
        PATH_TO_FROZEN_GRAPH = os.path.join(CWD_PATH,frozenModel)
    except FileNotFoundError:
        print("Wrong file or file path")

    # path to the label map
    labelMap="label_map_ppe1.pbtxt"
    try:
        PATH_TO_LABEL_MAP = os.path.join(CWD_PATH,labelMap)
    except FileNotFoundError:
        print("Wrong file or file path")
        
    # number of classes 
    NUM_CLASSES = 4

    # Load the label map.
    label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)

        
    # Label maps map indices to category names, so that when our detecting the object
    # the model will return a number, this number will given the object name based on the return value
    # it mapping integers to appropriate string labels 
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
        
    # video data source   
    cap=loadWebcam() #use webcam as video data source

    #start the detection by giving video data source
    start_detection(cap)
    
def if __name__ == "__main__":
    

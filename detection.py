# Importing packages necessary for the program to operate
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util
import RPi.GPIO as GPIO
from espeak import espeak

# Initializing connections for the GPIO pins on the Raspberry Pi
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(22,GPIO.OUT)
servo1 = GPIO.PWM(22,50)
pinTrigger = 18
pinEcho = 4
GPIO.setup(pinTrigger, GPIO.OUT)
GPIO.setup(pinEcho, GPIO.IN)
GPIO.output(pinTrigger, False)
time.sleep(0.5)

servo1.start(0)

# Define the VideoStream class in order to fetch the live video feed from the webcam
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the webcam and the camera image feed
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Retrieve/Read the first frame of the live webcam feed
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Reads the frames of the webcam video feed
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Enters a forever loop until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	    # Return the most recent frame
        return self.frame

    def stop(self):
	    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

# Defines the model, graph, and label files for the object detection model 
MODEL_NAME = 'ObjectModelList'
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Removes the ??? labels from the pre-prepared object detection model
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

# Enters a forever loop to create the video stream from the webcam
while True:
    # Grabs frame from live webcam video stream
    frame1 = videostream.read()

    # Gets the frame of the webcam video stream and resizes it
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Perform the actual detection by running the model with the image (frame) as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Model completes its evaluation of the objects in the image and retrieves the information
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Place dot in the center of the screen
            center_x = 640
            center_y = 360
            center_coordinates = (center_x,center_y)
            radius=3
            color = (0,0,255)
            thickness=-1
            cv2.circle(frame, center_coordinates, radius, color, thickness)

            #Put circle in the middle of the bounding box (this is to indicate whether it is to the right or left)
            xcenter = xmin + (int(round((xmax-xmin)/2)))
            ycenter = ymin + (int(round((ymax-ymin)/2)))
            cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)

            # Say if the object is to the left or right
            if (center_x > xcenter):
                direction = 0 # left
            else: 
                direction = 1 # right
            
            # Find the number of pixels between the center of the screen and the center of the bounding box in the x-direction
            pixel_difference_servo = abs(center_x - xcenter)
            angle_difference = pixel_difference_servo/10.5  
            
            # Rotates the servo motor according to the direction and angle as seen before
            if (direction == 0):
                angle = 90 + angle_difference
                
                # Converts to radians and rotates the servo to the necessary angle
                servo1.ChangeDutyCycle(2+(angle/18))
                time.sleep(0.5)
                servo1.ChangeDutyCycle(0)
                
            else:
                angle = 90 - angle_difference

                # Converts to radians and rotates the servo to the necessary angle
                servo1.ChangeDutyCycle(2+(angle/18))
                time.sleep(0.5)
                servo1.ChangeDutyCycle(0)

            # Distance Measurement Program
            # Initializes all of the GPIO pins of the Raspberry Pi           
            GPIO.setmode(GPIO.BCM)
            pinTrigger = 18
            pinEcho = 4
            GPIO.setup(pinTrigger, GPIO.OUT)
            GPIO.setup(pinEcho, GPIO.IN)
            GPIO.output(pinTrigger, False)
            time.sleep(0.5)
            GPIO.output(pinTrigger, True)
            time.sleep(0.0001)
            GPIO.output(pinTrigger, False)

            # Logs the pulse time, which is where the trigger pin pulses the ultrasonic sound wave
            while GPIO.input(pinEcho)==0:
                pulseStart = time.time()

            # Logs the pulse time, which is where the echo pin receives a reflected signal/ultrasonic sound wave
            while GPIO.input(pinEcho) == 1:
                pulseEnd = time.time()

            # Gets the time difference to find the distance
            pulseDuration = pulseEnd - pulseStart

            # Calculates the distance from the sensor to the object based on the time difference
            distance = ((pulseDuration * 17150)+1.15)
            distance = (distance/100)
            distance = round(distance)

            # Prints out the distance to the console
            print(("Distance is: ") + str(distance) + " meters")
            time.sleep(0.5)
            
            # F string literal to dictate the direction and distance of the object
            espeak.synth(f'There is an object to your {"left" if not direction else "right"} {distance} meters away')
            time.sleep(2)

    # Displays the live webcam feed to the Raspberry Pi OS
    cv2.imshow('Obstruction Watcher', frame)
        
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
GPIO.cleanup()
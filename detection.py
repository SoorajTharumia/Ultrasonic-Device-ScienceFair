# Import packages
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util
import RPi.GPIO as GPIO
import time
from espeak import espeak

# Sets up the GPIO devices and connections
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

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
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

MODEL_NAME = 'ObjectModelList'
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

while True:

    frame1 = videostream.read()

    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0] 
    classes = interpreter.get_tensor(output_details[1]['index'])[0] 
    scores = interpreter.get_tensor(output_details[2]['index'])[0] 
    
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            
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

            #Put circle in the middle (this is to indicate whether it is to the right or left)
            xcenter = xmin + (int(round((xmax-xmin)/2)))
            ycenter = ymin + (int(round((ymax-ymin)/2)))
            cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)

            # Say if the object is to the left or right
            if (center_x > xcenter):
                direction = 0 # left
            else: 
                direction = 1 # right
            
            # Calculate pixel difference in the screen and translate to an angle
            pixel_difference_servo = abs(center_x - xcenter)
            angle_difference = pixel_difference_servo/10.5  
            
            # Rotation of the servo motor holder
            if (direction == 0 ):
                angle = 90 + angle_difference
                
                servo1.ChangeDutyCycle(2+(angle/18))
                time.sleep(0.5)
                servo1.ChangeDutyCycle(0)  
            else:
                angle = 90 - angle_difference
                
                servo1.ChangeDutyCycle(2+(angle/18))
                time.sleep(0.5)
                servo1.ChangeDutyCycle(0)

            # Distance Measurement      
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

            while GPIO.input(pinEcho)==0:
                pulseStart = time.time()

            while GPIO.input(pinEcho) == 1:
                pulseEnd = time.time()

            pulseDuration = pulseEnd - pulseStart

            distance = ((pulseDuration * 17150)+1.15)
            distance = (distance/100)
            distance = round(distance)

            print(("Distance is: ") + str(distance) + " meters")
            
            time.sleep(0.5)
            
            # Text to Speech synthesis (tells the user where the object is)
            if (direction == 0) and (distance == 1):
                espeak.synth("There is an object to your left 1 meter away")
                time.sleep(2)
            elif (direction == 0) and (distance == 2):
                espeak.synth("There is an object to your left 2 meters away")
                time.sleep(2)
            elif (direction == 0) and (distance == 3):
                espeak.synth("There is an object to your left 3 meters away")
                time.sleep(2)
            elif (direction == 0) and (distance == 4):
                espeak.synth("There is an object to your left 4 meters away")
                time.sleep(2)
            elif (direction == 0) and (distance == 5):
                espeak.synth("There is an object to your left 5 meters away")
                time.sleep(2)
            elif (direction == 0) and (distance == 6):
                espeak.synth("There is an object to your left 6 meters away")
                time.sleep(2)
            elif (direction == 0) and (distance == 7):
                espeak.synth("There is an object to your left 7 meters away")
                time.sleep(2)
            elif (direction == 0) and (distance == 8):
                espeak.synth("There is an object to your left 8 meters away")
                time.sleep(2)
            elif (direction == 0) and (distance == 9):
                espeak.synth("There is an object to your left 9 meters away")
                time.sleep(2)
            elif (direction == 0) and (distance == 10):
                espeak.synth("There is an object to your left 10 meters away")
                time.sleep(2)
            elif (direction == 1) and (distance == 1):
                espeak.synth("There is an object to your right 1 meter away")
                time.sleep(2)
            elif (direction == 1) and (distance == 2):
                espeak.synth("There is an object to your right 2 meters away")
                time.sleep(2)
            elif (direction == 1) and (distance == 3):
                espeak.synth("There is an object to your right 3 meters away")
                time.sleep(2)
            elif (direction == 1) and (distance == 4):
                espeak.synth("There is an object to your right 4 meters away")
                time.sleep(2)
            elif (direction == 1) and (distance == 5):
                espeak.synth("There is an object to your right 5 meters away")
                time.sleep(2)
            elif (direction == 1) and (distance == 6):
                espeak.synth("There is an object to your right 6 meters away")
                time.sleep(2)
            elif (direction == 1) and (distance == 7):
                espeak.synth("There is an object to your right 7 meters away")
                time.sleep(2)
            elif (direction == 1) and (distance == 8):
                espeak.synth("There is an object to your right 8 meters away")
                time.sleep(2)
            elif (direction == 1) and (distance == 9):
                espeak.synth("There is an object to your right 9 meters away")
                time.sleep(2)
            elif (direction == 1) and (distance == 10):
                espeak.synth("There is an object to your right 10 meters away")
                time.sleep(2)

    cv2.imshow('Obstruction Watcher', frame)
        
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()
GPIO.cleanup()
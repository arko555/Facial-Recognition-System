from flask import Flask, render_template,render_template_string, Response
from camera import VideoCamera
import dill
import pickle
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from PIL import Image
from mtcnn.mtcnn import MTCNN
import mtcnn
from matplotlib import pyplot as plt
from numpy import expand_dims
import numpy as np
import tensorflow.keras.backend as K
from keras.models import load_model
import matplotlib.image as mpimg
import cv2
from scipy.spatial.distance import cosine
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)

##### Loading DB files ###########
pickle_in = open("recognition.pickle","rb")
classifier = dill.load(pickle_in)

data = np.load('train_embeddings.npz')
train_embeddings, filepaths, classes = data['arr_0'], data['arr_1'], data['arr_2']

file = open("train_labels",'rb')
class_indices = pickle.load(file)
file.close()

model = load_model('facenet_keras.h5')
#############################################

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')

def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/answer',methods=["Get"])
def recognition():
    
    answer = classifier('/home/arko/Documents/Python_Scripts/Capstone_Project/open_cv/opencv_frame_0.jpg')
    
    if answer != "please retake the image" and answer!= "Access Denied":
        answer = answer[0].upper() + answer[1:]
        code = "Hey! " + answer + " Access Granted"
        html = """\
               <html>
                 <head></head>
                 <body style="background-color:powderblue;">
                      <br><br><h1 style="color:green;">{code}</h1><br>
                 </body>
               </html>
               """.format(code=code)
        
        return render_template_string(html)
    
    elif answer == "please retake the image":
        code = answer[0].upper() + answer[1:]
        
        html = """\
               <html>
                 <head></head>
                 <body style="background-color:powderblue;">
                      <br><br><h1 style="color:yellow;">{code}</h1><br>
                 </body>
               </html>
               """.format(code=code)
        
        return render_template_string(html)
    
    else:
        code = answer
        
        html = """\
               <html>
                 <head></head>
                 <body style="background-color:powderblue;">
                      <br><br><h1 style="color:red;">{code}</h1><br>
                 </body>
               </html>
               """.format(code=code)
        
        return render_template_string(html)

if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)
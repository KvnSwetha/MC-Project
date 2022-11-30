import flask
import werkzeug
import base64
import io
import os
import time

from flask import jsonify
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from keras.models import load_model
from subprocess import check_output

app = flask.Flask(__name__)

@app.route('/check', methods=['GET'])
def check():
    return "Flask Server Working Successfully"

@app.route('/upload', methods = ['GET', 'POST'])
def handle_request():
    print("Request received")
    imagefile = flask.request.json['image']
    imageData = base64.b64decode(imagefile)
    
    imageArray = np.fromstring(imageData, np.uint8)
    image = cv2.imdecode(imageArray, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28,28), interpolation=cv2.INTER_NEAREST)
    res = Image.fromarray(image)
    res = np.array(res).flatten()
    # res = np.array(res.resize((28,28)))

    blackThresh = 128
    numOfBlack = 0
    for pixel in res:
        if pixel < blackThresh:
            numOfBlack += 1
    size = len(res)
    
    if numOfBlack/float(size) < 0.5:
        _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
        res = thresh
    else:
        _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        res = thresh

    # res = np.reshape(res,[-1,784])
    res = res.astype('float32')/255
    res = res.reshape(-1, 28, 28, 1)
 
    trainedModel = load_model('cnnModel')
    pred = trainedModel.predict(res)
    category = np.argmax(pred, axis=None, out=None)
    print(category)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = 'Img_'+timestr
    imagePath = (str(category)+"/"+filename)
    os.makedirs(os.path.dirname(imagePath), exist_ok=True)
    img = Image.open(io.BytesIO(imageData))
    img.save(imagePath, 'png')
    return str(category)

app.run(host="0.0.0.0", port=8081, debug=True)

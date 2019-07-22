from flask import Flask, request
from flask.json import jsonify
from flask import json
import os
import base64
import numpy as np
import cv2
import time

app=Flask(__name__)


#欢迎界面
@app.route('/',methods=["GET","POST"])
def helloWorld():
    return jsonify({'qx':'hello world'})

#欢迎界面
@app.route('/sortvideo',methods=["GET","POST"])
def sortvideo():
    print(request)
    # print(request.data['img1'])
    # print(request.json['time'])
    print("request Data")
    print(request.data)
    upload_file = request.files['img0']
    old_file_name = upload_file.filename
    if upload_file:
        file_path = os.path.join('./images/', old_file_name+'.jpg')
        upload_file.save(file_path)

    return jsonify('上传成功')





if __name__=='__main__':

    app.run(debug=False)
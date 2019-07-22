from flask import Flask, request
from flask.json import jsonify
from flask import json


app=Flask(__name__)



#欢迎界面
@app.route('/',methods=["GET","POST"])
def helloWorld():
    return jsonify({'qx':'hello world'})

#欢迎界面
@app.route('/sortvideo',methods=["GET","POST"])
def sortvideo():
    return jsonify('sortvideo')





if __name__=='__main__':
    app.run(debug=False)
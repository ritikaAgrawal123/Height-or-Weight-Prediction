from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)
modelw=pickle.load(open('model.pkl','rb'))
modelh=pickle.load(open('modelh.pkl','rb'))

@app.route("/")
def hello():
    return render_template("index.html")

@app.route('/home',methods=['POST','GET'])
def home():
    data1 =int(request.form.get('height'))
    gender=int(request.form.get('gender'))
    arr = np.array([gender,data1]).reshape(1,-1)
    prediction = modelw.predict(arr)
    output='{0:.{1}f} pounds'.format(prediction[0], 2)
    return render_template('index.html',weight=output)
@app.route('/home1',methods=['POST','GET'])
def home1():
    data1 =int(request.form.get('weight'))
    gender=int(request.form.get('gender'))
    arr = np.array([gender,data1]).reshape(1,-1)
    prediction = modelh.predict(arr)
    output='{0:.{1}f} inches'.format(prediction[0], 2)
    return render_template('index.html',height=output)
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route("/")
def hello():
    return render_template("index.html")

@app.route('/home',methods=['POST','GET'])
def home():
    data1 =int(request.form.get('height'))
    gender=int(request.form.get('gender'))
    arr = np.array([gender,data1]).reshape(1,-1)
    prediction = model.predict(arr)
    return render_template('index.html',weight=prediction)
@app.route('/home1',methods=['POST','GET'])
def home1():
    data1 =int(request.form.get('weight'))
    gender=int(request.form.get('gender'))
    arr = np.array([gender,data1]).reshape(1,-1)
    prediction = model.predict(arr)
    return render_template('index.html',height=prediction)
if __name__ == '__main__':
    app.run(debug=True)

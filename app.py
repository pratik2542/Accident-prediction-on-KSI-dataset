
from urllib import request
import joblib as jb
from flask import Flask, request, jsonify, render_template
import traceback
import pandas as pd
import json
import sys
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn import *

app = Flask(__name__)
ap = ""
name = ""
y_train =jb.load('y_train.pkl')
y_test = jb.load('y_test.pkl')
y_test1=""
accuracy = ""
y_train_predict = ""
@app.route("/", methods=["GET", "POST"])
def Fun_knn():
    return render_template("index.html")


@app.route("/sub", methods=["GET", "POST"])
def submit():
    if request.method == "POST":
        input_dict = request.form.to_dict()
        model = input_dict.pop('model')
        int_features = [int(x) for x in request.form.values()]
        final = [np.array(int_features)]
        model = final[0].any()
        print(int_features)
        print(final)
        int_features = pd.DataFrame.from_dict([input_dict])
        

        if model == 'knn':
            loaded_model = jb.load('KNN.pkl')
            y_test1=jb.load('y_test.pkl')
            y_train_predict = jb.load('y_pred_knn.pkl')
            predictions = loaded_model.predict(final)                   
            accuracy = accuracy_score(y_test1,y_train_predict)
            output = predictions[0]
            print('Accuracy KNN: ',accuracy)
            if output == 0:
                return render_template("sub.html", prediction_text="Injury is fatal", accuracy = accuracy,model = model)
            else:
                return render_template("sub.html", prediction_text="Injury is non fatal", accuracy = accuracy,model = model)

        elif model == 'svm':
            loaded_model = jb.load('svm.pkl')
            y_train_predict = jb.load('y_test_pred_svm.pkl')
            predictions = loaded_model.predict(final).toArray()
            accuracy = accuracy_score(y_train,y_train_predict)
            output = predictions[0]
            print('Accuracy SVM: ',accuracy)
            if output == 0:
                return render_template("sub.html", prediction_text="Injury is fatal",accuracy = accuracy,model = model)
            else:
                return render_template("sub.html", prediction_text="Injury is non fatal", accuracy = accuracy,model = model)

        elif model == 'nn':
            loaded_model = jb.load('NN.pkl')
            predictions = loaded_model.predict(final).tolist()
            output = predictions[0]
            y_train_predict = jb.load('predictions_nn.pkl')
            accuracy = accuracy_score(y_test,y_train_predict)
            if output == 0:
                return render_template("sub.html", prediction_text="Injury is fatal", accuracy = accuracy,model = model)
            else:
                return render_template("sub.html", prediction_text="Injury is non fatal", accuracy = accuracy,model = model)
        elif model == 'dt':
            loaded_model = jb.load('dt.pkl')
            y_train_predict = jb.load('clf_tree_pred.pkl')
            predictions = loaded_model.predict(final).tolist()
            
            output = predictions[0]
            accuracy = accuracy_score(y_test,y_train_predict)
            if output == 0:
                return render_template("sub.html", prediction_text="Injury is fatal", accuracy = accuracy,model = model)
            else:
                return render_template("sub.html", prediction_text="Injury is non fatal", accuracy = accuracy,model = model)
        elif model == 'rf':
            loaded_model = jb.load('Rf.pkl')
            
            y_train_predict = jb.load('rf_y_pred_rf.pkl')
            predictions = loaded_model.predict(final).tolist()
            
            output = predictions[0]
            accuracy = accuracy_score(y_test,y_train_predict)
            
            if output == 0:
                return render_template("sub.html", prediction_text="Injury is fatal", accuracy = accuracy,model = model)
            else:
                return render_template("sub.html", prediction_text="Injury is non fatal",accuracy = accuracy,model = model)
        elif model == 'lr':
            loaded_model = jb.load('lr.pkl')
            y_test1= jb.load('y_test.pkl')
            y_train_predict = jb.load('y_test_pred_lr.pkl')
            predictions = loaded_model.predict(final).tolist()
            #print('Accuracy', accuracy_score(y_test, y_grid_pred))
            output = predictions[0]
            accuracy = accuracy_score(y_test1,y_train_predict)
            output = predictions[0]
            if output == 0:
                return render_template("sub.html", prediction_text="Injury is fatal", accuracy = accuracy,model = model)
            else:
                return render_template("sub.html", prediction_text="Injury is non fatal", accuracy = accuracy,model = model)

if __name__=="__main__":
    app.run(debug=True) 

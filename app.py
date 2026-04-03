from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sklearn
import pickle
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','b'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potessium'])
    temp = float(request.form['Temprature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)

    prediction = model.predict(sc_mx_features)

    crop_dict = {
        1: "rice", 2: "maize", 3: "jute", 4: "cotton", 5: "coconut",
        6: "papaya", 7: "orange", 8: "apple", 9: "muskmelon",
        10: "watermelon", 11: "grapes", 12: "mango", 13: "banana",
        14: "pomegranate", 15: "lentil", 16: "blackgram",
        17: "mungbeans", 18: "mothbeans", 19: "pigeonpeas",
        20: "kidneybeans", 21: "chickpea", 22: "coffee"
    }


    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop."

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
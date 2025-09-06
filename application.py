from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# import model and scale

model = pickle.load(open('Model.pkl','rb'))
scaler = pickle.load(open("Scaler.pkl",'rb'))

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict_data",methods = ['GET',"POST"])
def predict_data():
    if request.method =='POST':
        Temperature = float(request.form.get("temperature"))
        RH          = float(request.form.get("rh"))
        Ws          = float(request.form.get("ws"))
        Rain        = float(request.form.get("rain"))
        FFMC        = float(request.form.get("ffmc"))
        DMC         = float(request.form.get("dmc"))
        ISI         = float(request.form.get("isi"))
        Classes     = float(request.form.get("classes"))
        Region      = float(request.form.get("region"))

        df = pd.DataFrame(data=[[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]], columns=scaler.feature_names_in_)
        print(df)
        new_data_scaled = scaler.transform(df)
        result = model.predict(new_data_scaled)

        print(new_data_scaled)
        return render_template("home.html",results =np.round( result[0],2))
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
import numpy as np
import pandas as pd
from flask import Flask,render_template,request
from flask_cors import CORS,cross_origin
import pickle


predictor=pickle.load(open('model.pkl','rb'))

app=Flask(__name__)
CORS(app)

data=pd.read_csv('cleaned_data.csv')

@app.route('/')
def home():
    companies=sorted(data['company'].unique())
    models=sorted(data['name'].unique())
    years=sorted(data['year'].unique())
    fuels=sorted(data['fuel_type'].unique())
    return render_template('index.html',companies=companies,models=models,years=years,fuels=fuels)


@app.route('/predict', methods=["POST"])
@cross_origin()
def predict():
    company=request.form.get('company')
    model=request.form.get('model')
    year=request.form.get('year')
    driven=request.form.get('driven')
    fuel=request.form.get('fuel')

    if company==None or model==None or year==None or driven==None or fuel==None:
        return str(company)+str(model)+str(year)+str(driven)+str(fuel)

    return "Prediction ₹"+str(round(predictor.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array([model,company,year,driven,fuel]).reshape(1,-1)))[0]  , 2))



if __name__=='__main__':
    app.run(debug=True)
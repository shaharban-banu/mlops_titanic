from fastapi import FastAPI
import pickle
import logging
from prometheus_fastapi_instrumentator import Instrumentator
import pandas as pd
import os


logging.basicConfig(filename="app.log",level=logging.INFO)

app=FastAPI()

Instrumentator().instrument(app).expose(app)

model=pickle.load(open("model.pkl","rb"))

LOG_FILE="data/current.csv"

@app.post('/predict')
def predict(Pclass:int,Sex:int,Fare:float,Age:float):
    pred=model.predict([[Pclass,Sex,Fare,Age]])
    logging.info(f"input : {[Pclass,Sex,Fare,Age]}  output :{pred[0]}")

    new_data=pd.DataFrame([
        {
            "Pclass":Pclass,
            "Sex":Sex,
            "Age":Age,
            "Fare":Fare
        }
    ])

    if not os.path.exists(LOG_FILE):
        new_data.to_csv(LOG_FILE,index=False)
    else:
        new_data.to_csv(LOG_FILE,mode='a',header=False,index=False)
    return {'Survived':int(pred[0])}
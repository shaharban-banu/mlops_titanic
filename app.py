from fastapi import FastAPI
import pickle
import logging

logging.basicConfig(filename="app.log",level=logging.INFO)

app=FastAPI()

model=pickle.load(open("model.pkl","rb"))

@app.post('/predict')
def predict(Pclass:int,Sex:int,Fare:float,Age:float):
    pred=model.predict([[Pclass,Sex,Age,Fare]])
    logging.info(f"input : {[Pclass,Sex,Age,Fare]}  output :{pred[0]}")
    return {'prediction':int(pred[0])}
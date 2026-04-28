import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment("titanic-experiment")


df=pd.read_csv("data/titanic.csv")
x=df.drop("Survived",axis=1)
y=df['Survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


with mlflow.start_run():
    n_estimators=100
    max_depth=5
    model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    acc=accuracy_score(y_test,pred)

    mlflow.log_param("n_estimatorss",n_estimators)
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_metric("accuracy",acc)
    with open("model.pkl","wb") as f:
        pickle.dump(model,f)
print("mlflow logging done")
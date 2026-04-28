import pandas as pd

df=pd.read_csv("data/train.csv")
print(df.head())
df=df[['Pclass','Sex','Age','Fare','Survived']]
df['Age']=df['Age'].fillna(df['Age'].median())
df['Fare']=df['Fare'].fillna(df['Fare'].median())
df['Sex']=df['Sex'].map({'male':0,'female':1})
print(df.head())
df.to_csv("data/titanic.csv",index=False)
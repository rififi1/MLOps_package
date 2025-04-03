from fastapi import FastAPI
import uvicorn
import pandas as pd
import joblib
import numpy as np
import os
from pydantic import BaseModel
#import glob

from scripts.data import Dataset
from scripts.model import Model

model_ = None

app = FastAPI()


"""def list_pkl_files(directory):
    # Construct the search pattern
    search_pattern = os.path.join(directory, '*.pkl')

    # Use glob to find all .pkl files
    pkl_files = glob.glob(search_pattern)

    return pkl_files"""

@app.on_event("startup")
async def load_model():
    print("on startup")
    print("Server is starting up.")
    
    path = "data"
    train_data = Dataset()
    train_data.load_data(os.path.join(path,'train.csv'))
    #df_test = Dataset(os.path.join(path, 'test.csv'))

    train_data.clean()
    #df_test.clean()

    train_features,train_labels = train_data.split_label_features() 

    global model_
    # TODO: add a check for existing pkl files, and load pkl if it exits
    file_name = "default"
    if os.path.isfile(f"{file_name}.pkl"):
        model_ = Model(pickle_name=file_name)
    else:
        model_ =  Model()
        model_.fit(train_features,train_labels, save_model=file_name)
    model_.set_accuracy(train_features,train_labels)
    print("model did fit, with accuracy ", model_.get_accuracy())  


@app.get("/")
async def root():
    return {"message": "Welcome to our MLOps API."}

@app.get("/accuracy")
async def getter_accuracy():
    global model_
    if model_ is None:
        return {"error": "Model not loaded"}
    try:
        accuracy = model_.get_accuracy()
        return {"prediction": accuracy}
    except Exception as e:
        return {"error": str(e)}

@app.get("/feature_order")
async def feature_order():
    path = "data/"
    df = Dataset(path+'test.csv')
    df.clean()
    return {"features in order:": str(df.get_features()),
            "types" : str(df.get_features_types())
            }

class Item(BaseModel):
    pclass: int
    sex: int
    age: float
    sibsp: int 
    parch: int
    fare: float
    embarked: int

class Chill(BaseModel):
    features: list


@app.post("/predict_one")
async def predict_one(chill: Chill):
    global model_
    
    if model_ is None:
        return {"error": "Model not loaded"}
    try:
        lst = list(chill.features)
        df = pd.DataFrame([lst])
        df.columns =['Pclass','Sex','Age',"SibSp","Parch","Fare","Embarked"]
        
        try:
            prediction = model_.predict(df)
        except Exception as e:
            raise e
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    
    uvicorn.run(app)
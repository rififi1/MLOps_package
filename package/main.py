from fastapi import FastAPI
import uvicorn
import pandas as pd
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
    """print("cwd: ", os.getcwd())
    print("list dir 0: ", os.listdir())"""
    
    models_path = "package/models"
    path = "package/data"
    train_data = Dataset()
    train_data.load_data(os.path.join(path,'train.csv'))
    #df_test = Dataset(os.path.join(path, 'test.csv'))

    train_data.clean()
    #df_test.clean()

    train_features,train_labels = train_data.split_label_features() 

    global model_
    # TODO: add a check for existing pkl files, and load pkl if it exits
    file_name = "default"
    print("trying to get model: ", os.path.join(models_path, f"{file_name}.pkl"))
    if os.path.isfile(os.path.join(models_path, f"{file_name}.pkl")):
        model_ = Model(pickle_name=file_name, models_folder=models_path)
        print("loaded existing default model")
    else:
        model_ =  Model(models_folder=models_path)
        model_.fit(train_features,train_labels, save_model=file_name)
        print("trained new default model")
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
    path = "package/data"
    df = Dataset()
    df.load_data(os.path.join(path,'test.csv'))
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


@app.get("/predict_one")
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
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
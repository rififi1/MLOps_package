from fastapi import FastAPI
import uvicorn
import pandas as pd
import joblib
import numpy as np

import scripts.data as data
import scripts.model as model

model_ = None

app = FastAPI()

@app.on_event("startup")
async def load_model():
    print("on startup")
    print("Server is starting up.")
    path = "data/"
    df_train = data.Data(path+'train.csv')
    df_test = data.Data(path+'test.csv')

    df_train.clean()
    df_test.clean()

    train_features,train_labels = df_train.split_label_features() 

    global model_
    model_ =  model.Model()
    model_.fit(train_features,train_labels)
    model_.set_accuracy(train_features,train_labels)
    print("model did fit, with accuracy ", model_.get_accuracy())  


@app.get("/")
async def root():
    return {"message": "Hello World"}

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
    df = data.Data(path+'test.csv')
    df.clean()
    return {"features in order:": str(df.get_features()),
            "types" : str(df.get_features_types())
            }

@app.post("/predict/")
async def predict(features: list[float]):
    global model_
    if model_ is None:
        return {"error": "Model not loaded"}
    try:
        # Ensure features are in the correct format (e.g., 2D array for scikit-learn)
        input_data = np.array(features).reshape(1, -1)
        prediction = model_.predict(input_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    
    uvicorn.run(app)
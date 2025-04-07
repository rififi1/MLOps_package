from fastapi import FastAPI
import uvicorn
import pandas as pd
import os

from scripts.data import Dataset
from scripts.model import Model
from scripts.utils import *
from objects.http_objects import ModelChoice, Features, Predictions

model_ = None

MODELS_PATH = "models"
DATA_PATH = "data"

MODELS = ['SVM', 'Logistic']

app = FastAPI()

@app.on_event("startup")
async def load_model():
    """When server is booting, load the model (or train it if no model is stored)."""

    print("on startup")
    print("Server is starting up.")
    
    # IMPORT AND CLEAN THE TRAIN DATASET

    train_data = Dataset()
    train_data.load_data(os.path.join(DATA_PATH,'train.csv'))
    train_data.clean()
    train_features,train_labels = train_data.split_label_features() # extract labels and features from the dataset
    

    # GET/TRAIN THE MODELS

    print("existing models: ", os.listdir(MODELS_PATH))
    train_all_models(train_features, train_labels, MODELS_PATH, MODELS)


    # LOAD THE 1ST MODEL AS DEFAULT MODEL

    default_model_type = MODELS[0] # 1st model is the default model
    print("trying to get model: ", os.path.join(MODELS_PATH, f"{default_model_type}.pkl"))

    global model_
    model_ = Model(models_folder=MODELS_PATH, model_type=default_model_type, pickle_=True)
    print("loaded default model", default_model_type)


    # SET AND GET MODEL'S ACCURACY

    model_.set_accuracy(train_features,train_labels)
    print("Using default model, ", default_model_type, ", with accuracy ", model_.get_accuracy())

@app.get("/")
async def root():
    """Basic route to check that the server is up and running."""
    return {"message": "Welcome to our MLOps API."}

@app.get("/accuracy")
async def getter_accuracy():
    """Return the model's accuracy."""
    global model_
    if model_ is None:
        return {"error": "Model not loaded"}
    try:
        accuracy = model_.get_accuracy()
        return {"prediction": accuracy}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/list_model")
async def list_model():
    """Return the list of trained models."""
    return{"ok": MODELS}

@app.put("/switch_model")
async def switch_model(model_choice: ModelChoice):
    global model_
    try:
        model_ = Model(pickle_=True, model_type=model_choice.model,models_folder=MODELS_PATH)
    except Exception as e:
            return {"error": str(e)}
    return {"ok": model_.model_type}

@app.get("/feature_order")
async def feature_order():
    """Shows features order and types. Most likely useless in prod setup, but very useful for dev."""

    df = Dataset()
    df.load_data(os.path.join(DATA_PATH,'test.csv'))
    df.clean()
    return {
        "features in order:": str(df.get_features()),
        "types" : str(df.get_features_types())
        }

@app.get("/predict")
async def predict_one(features: Features):
    """Predict if a person will survive given their features."""

    global model_
    
    if model_ is None:
        return {"error": "Model not loaded"}
    try:
        # FORMAT FEATURES 
        df = pd.DataFrame.from_dict(features)

        # MODEL PREDICTION
        try:
            prediction = model_.predict(df)
            prediction = Predictions(pred=list(prediction))

        except Exception as e:
            return {"error": str(e)}
            #raise e
        return {"prediction": prediction.pred}
    
    except Exception as e:
        # Normally the server does not return the error, but here in a dev environment it does make sense.
        return {"error": str(e)}

if __name__ == "__main__":
    # expose the docker VM to the computer's ports
    uvicorn.run(app)#, host="0.0.0.0", port=8000) # uncomment when using docker  # when not, comment it or you'll expose your machine

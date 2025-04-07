from fastapi import FastAPI
import uvicorn
import pandas as pd
import os

from scripts.data import Dataset
from scripts.model import Model
from scripts.utils import *
from objects.http_objects import ModelChoice, Features, Predictions

model_ = None

app = FastAPI()

@app.on_event("startup")
async def load_model():
    """When server is booting, load the model (or train it if no model is stored)."""

    print("on startup")
    print("Server is starting up.")
    
    # IMPORT AND CLEAN THE TRAIN DATASET

    path = "data"
    train_data = Dataset()
    train_data.load_data(os.path.join(path,'train.csv'))
    train_data.clean()
    train_features,train_labels = train_data.split_label_features() # extract labels and features from the dataset
    

    # GET/TRAIN THE MODELS

    models_folder = "models"
    print("existing models: ", os.listdir(models_folder))

    models = ['SVM', 'Logistic']
    train_all_models(train_features, train_labels, models_folder, models)


    # LOAD THE 1ST MODEL AS DEFAULT MODEL

    default_model_type = models[0] # 1st model is the default model
    print("trying to get model: ", os.path.join(models_folder, f"{default_model_type}.pkl"))

    global model_
    model_ = Model(models_folder=models_folder, model_type=default_model_type, pickle_=True)
    print("loaded default model", default_model_type)


    # SET AND GET MODEL'S ACCURACY

    model_.set_accuracy(train_features,train_labels)
    print("Using default model, ", default_model_type, ", with accuracy ", model_.get_accuracy())

    """if os.path.isfile(os.path.join(models_folder, f"{file_name}.pkl")):
        # If default model is stored in a pickle file: instantiate a model with the existing pickle file 
        model_ = Model(pickle_name=file_name, models_folder=models_folder)
        print("loaded existing default model")
    else:
        # Else, train our own default model (and save it as default)
        model_ =  Model(models_folder=models_folder)
        model_.fit(train_features,train_labels, save_model=file_name)
        print("trained new default model")
    """


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

@app.put("/switch_model")
async def switch_model(model_choice: ModelChoice):
    global model_
    models_folder="models"
    try:
        model_ = Model(pickle_=True, model_type=model_choice.model,models_folder=models_folder)
    except Exception as e:
            return {"error": e}
    return {"ok": model_.model_type}


@app.get("/feature_order")
async def feature_order():
    """Shows features order and types. Most likely useless in prod setup, but very useful for dev."""

    path = "data"
    df = Dataset()
    df.load_data(os.path.join(path,'test.csv'))
    df.clean()
    return {"features in order:": str(df.get_features()),
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
    uvicorn.run(app) #, host="0.0.0.0", port=8000) # uncomment when using docker # otherwise comment it or you'll expose your machine
    
from fastapi import FastAPI
import uvicorn
import pandas as pd
import os

from scripts.data import Dataset
from scripts.model import Model
from scripts.utils import *
from scripts.api_helper import *
from objects.http_objects import ModelChoice, Features, Predict, Predictions

#model_ = None

MODELS_PATH = "models"
DATA_PATH = "data"
ACCURACY_FILE_NAME = "accuracies.csv"

MODELS = ['SVM', 'Logistic']
DEFAULT_MODEL = MODELS[0]

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
    

    # TRAIN THE MODELS

    print("existing models: ", os.listdir(MODELS_PATH))
    train_all_models(train_features, train_labels, MODELS_PATH, MODELS, ACCURACY_FILE_NAME)


@app.get("/")
async def root():
    """Basic route to check that the server is up and running."""
    return {"message": "Welcome to our MLOps Titanic API."}

# DEPRECATED, use /model/current or /model/list instead 
@app.get("/accuracy")
async def getter_accuracy():
    """Return the model's accuracy."""
    global model_
    if model_ is None:
        return {
            "error": "Model not loaded",
            "WARNING": "route is deprecated, use /model/current or /model/list instead."
            }
    try:
        accuracy = model_.get_accuracy()
        return {
            "prediction": accuracy,
            "WARNING": "route is deprecated, use /model/current or /model/list instead."
            }
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/model/default")
async def default_model():
    """Return the default model's name and accuracy."""
    
    try:
        model_ = Model(models_folder=MODELS_PATH, model_type=DEFAULT_MODEL, pickle_=True)
        model_.load_accuracy(os.path.join(MODELS_PATH, ACCURACY_FILE_NAME))
        accuracy = model_.get_accuracy()
        return {
            "model": model_.model_type, # equivalent to DEFAULT_MODEL
            "accuracy": accuracy
            }
    except Exception as e:
        return {"error": str(e)}

@app.get("/model/list")
async def list_model():
    """Return the list of trained and ready-to-use models."""
    accuracies = get_accuracies(os.path.join(MODELS_PATH, ACCURACY_FILE_NAME))
    print(accuracies)
    accuracies = accuracies.T.to_dict('dict')
    print(accuracies)
    return accuracies

# DEPRECATED: a REST API shouldn't store data between requests. Use the model parameter in /predict/ instead.
@app.put("/model/switch")
async def switch_model(model_choice: ModelChoice):
    global model_
    try:
        model_ = Model(pickle_=True, model_type=model_choice.model,models_folder=MODELS_PATH)
        model_.load_accuracy(os.path.join(MODELS_PATH, ACCURACY_FILE_NAME))
    except Exception as e:
            return {"error": str(e)}
    return {
        "ok": model_.model_type,
        "WARNING": "Route is deprecated. You can chose your model direclty in the /predict/ route. "
        }

@app.get("/features")
async def features():
    """Shows features order and types. Most likely useless in prod setup, but very useful for dev."""

    df = Dataset()
    df.load_data(os.path.join(DATA_PATH,'test.csv'))
    df.clean()
    return {
        "features in order": str(df.get_features()),
        "types" : str(df.get_features_types())
        }

@app.get("/predict")
async def predict(predict: Predict):
    """Predict if a person will survive given their features."""

    try:
        # FORMAT FEATURES 
        df = pd.DataFrame.from_dict(predict.features)

        # MODEL PREDICTION
        try:
            model_ = Model(pickle_=True, model_type=predict.model,models_folder=MODELS_PATH)
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

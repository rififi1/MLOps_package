from pydantic import BaseModel
from typing_extensions import TypedDict

class ModelChoice(BaseModel):
    model:str

class Features(TypedDict):
    Pclass: list[int]
    Sex: list[int]
    Age: list[float]
    SibSp: list[int]
    Parch: list[int]
    Fare: list[float]
    Embarked: list[int]

class Predict(BaseModel):
    model: str
    features: Features

class Predictions(BaseModel):
    pred: list[int]

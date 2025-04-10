from pydantic import BaseModel
from typing_extensions import TypedDict
from typing import List

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

class Predictions(BaseModel):
    pred: list[int]

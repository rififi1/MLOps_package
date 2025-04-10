import os
import pandas as pd

from scripts.model import Model
from scripts.utils import *

def train_all_models(train_features, train_labels, models_folder, models, accuracy_file_name):
    accuracies_path = os.path.join(models_folder, accuracy_file_name)
    if os.path.isfile(accuracies_path):
        accuracies = get_accuracies(accuracies_path)
        print(accuracies)
    else:
        accuracies = pd.DataFrame(columns=['model', 'accuracy']).set_index("model", drop=True)
    for model in models:
        if os.path.isfile(os.path.join(models_folder, f"{model}.pkl")):
            # If model already exists, don't retrain it
            continue

        # train new model 
        m = Model(models_folder=models_folder, model_type=model)
        m.fit(train_features,train_labels, save_model=model)
        m.set_accuracy(train_features,train_labels)

        # TODO: it would make more sense to put that behaviour in the Model class (in set_accuracy, or even a new save_accuracy method)
        #accuracies.concat([[model, m.get_accuracy()]])
        accuracies.loc[model, "accuracy"] = m.get_accuracy()

        print("trained model " , model, " with accuracy: ", m.get_accuracy())
        print("_ _ _ _")
    accuracies.to_csv(accuracies_path, header=True, index_label="model")

import os
from scripts.model import Model


def train_all_models(train_features, train_labels, models_folder, models):
    for model in models:
        if os.path.isfile(os.path.join(models_folder, f"{model}.pkl")):
            # If model already exists, don't retrain it
            continue

        # train new model 
        m = Model(models_folder=models_folder, model_type=model)
        m.fit(train_features,train_labels, save_model=model)
        m.set_accuracy(train_features,train_labels)

        print("trained model " , model, " with accuracy: ", m.get_accuracy())
        print("_ _ _ _")
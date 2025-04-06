from sklearn.svm import SVC
import pickle
import os

class Model():
    
    def __init__(self, pickle_name=None, models_folder="models"):
        self.models_folder = models_folder
        print("models folder (existing models): ", os.listdir(models_folder))

        if pickle_name is None:
            self.model = SVC()
        else:
            with open(os.path.join(self.models_folder, f"{pickle_name}.pkl"), "rb") as f:
                self.model = pickle.load(f)
        self.accuracy = None
    
    def fit(self,train_features,train_labels, save_model=None):
        self.model.fit(train_features,train_labels)

        if save_model:
            with open(os.path.join(self.models_folder, f"{save_model}.pkl"), "wb") as f:
                pickle.dump(self.model, f)
        
    def set_accuracy(self,test_features,test_labels): 
        self.accuracy = self.model.score(test_features,test_labels)
    
    def get_accuracy(self):
        return self.accuracy
    
    def predict(self, features):
        return self.model.predict(features)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle
import os

class Model():
    
    def __init__(self, model_type='SVM', models_folder="models", pickle_=False):
        self.models_folder = models_folder
        self.model_type = model_type

        assert model_type in ['SVM', 'Logistic'], "model_type should be 'SVM' or 'Logistic'"

        if pickle_ == False:
            # INSTANCIATE A MODEL

            if model_type == 'SVM':
                self.model = SVC()
            elif model_type == 'Logistic':
                self.model = LogisticRegression()

        else:
            # LOAD A MODEL
            pickle_file = os.path.join(self.models_folder, f"{model_type}.pkl")
            assert os.path.isfile(pickle_file), "could not load the required model"
            with open(pickle_file, "rb") as f:
                self.model = pickle.load(f)
            # TODO: store accuracy in a file and extract it here

        self.accuracy = None
    
    def fit(self,train_features,train_labels, save_model=None):
        """Train and save the model."""

        # TRAIN THE MODEL
        self.model.fit(train_features,train_labels)

        # SAVE MODEL IN A PICKLE FILE IF REQUIRED
        if save_model:
            with open(os.path.join(self.models_folder, f"{save_model}.pkl"), "wb") as f:
                pickle.dump(self.model, f)
        
    def set_accuracy(self,test_features,test_labels): 
        """Set model's accuracy."""
        self.accuracy = self.model.score(test_features,test_labels) # SVC's default score is its accuracy.
    
    def get_accuracy(self):
        """Get model's accuracy."""
        return self.accuracy
    
    def predict(self, features):
        """Predict suing the model."""
        return self.model.predict(features)
    
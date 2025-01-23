from sklearn.svm import SVC

class Model():
    
    def __init__(self):
        self.model = SVC()
        self.accuracy = None
    
    def fit(self,train_features,train_labels):
        self.model.fit(train_features,train_labels)

    def set_accuracy(self,test_features,test_labels): 
        self.accuracy = self.model.score(test_features,test_labels)
    
    def get_accuracy(self):
        return self.accuracy
    
    def predict(self, features):
        return self.model.predict(features)
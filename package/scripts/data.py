import pandas as pd

class Dataset():
    def __init__(self):
        self.df_cleaned = None
        self.my_sex_dict = {
            'male': 0,
            'female': 1
            }
        self.my_embarked_dict = {
            'S': 0,
            'C': 1,
            'Q': 2
            }

    def load_data(self, path):
        self.df = pd.read_csv(path)

    def clean(self):
        try:
            self.df_cleaned = self.df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
            self.df_cleaned = self.df_cleaned.dropna(subset=['Embarked','Age'])

            
            self.df_cleaned['Sex'] = self.df_cleaned['Sex'].apply(lambda x: self.my_sex_dict[x])          

            self.df_cleaned['Embarked'] = self.df_cleaned['Embarked'].apply(lambda x: self.my_embarked_dict[x])
        except Exception as e:
            raise e

        return self.df_cleaned

    def split_label_features(self):
        labels = self.df_cleaned['Survived']
        features = self.df_cleaned.drop(columns='Survived')
        return features,labels
    
    def get_features(self):
        return str(list(self.df_cleaned.columns.values))
    
    def get_features_types(self):
        return str(list(self.df_cleaned.dtypes))

    def __len__(self):
        return self.df_cleaned.shape[0]
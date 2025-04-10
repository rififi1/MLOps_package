import pandas as pd

class Dataset():
    def __init__(self):
        self.df_cleaned = None

        # Label encoding maps
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
        """Load data given a path."""
        self.df = pd.read_csv(path)

    def clean(self):
        """Clean the dataset."""
        try:
            # Drop some columns
            self.df_cleaned = self.df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
            # Drop null values
            self.df_cleaned = self.df_cleaned.dropna(subset=['Embarked','Age'])

            # Label encoding the Sex (gender) attribute.
            self.df_cleaned['Sex'] = self.df_cleaned['Sex'].apply(lambda x: self.my_sex_dict[x])    
            # Label encoding the Embarked attribute. 
            self.df_cleaned['Embarked'] = self.df_cleaned['Embarked'].apply(lambda x: self.my_embarked_dict[x])
        except Exception as e:
            raise e

        return self.df_cleaned

    def split_label_features(self):
        """Extract labels and features from the dataset."""
        labels = self.df_cleaned['Survived']
        features = self.df_cleaned.drop(columns='Survived')
        return features,labels
    
    def get_features(self):
        """Return the list of feature names."""
        return str(list(self.df_cleaned.columns.values))
    
    def get_features_types(self):
        """Return the list of feature types."""
        return str(list(self.df_cleaned.dtypes))

    def __len__(self):
        """Returns length of the dataset."""
        return self.df_cleaned.shape[0]
    
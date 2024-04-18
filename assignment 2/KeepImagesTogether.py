from sklearn.model_selection import  train_test_split
import pandas as pd
from ast import literal_eval
import os
from sklearn.preprocessing import LabelEncoder

class KeepImagesTogether:
    def __init__(self):
        df = pd.read_csv("dataset.csv")
        df = df[['sentiment', 'screenshots']]
        df = df.dropna(subset = 'sentiment')
        self.label_encoder = LabelEncoder()
        df['sentiment'] = self.label_encoder.fit_transform(df['sentiment'])
        self.training_data = df.sample(frac = 0.7, random_state=123)
        self.test_data = df.drop(self.training_data.index)

    def get_training_set(self):
        return self.training_data
    
    def get_test_set(self):
        return self.test_data
    
    def get_original_label(self, label):
        return self.label_encoder.inverse_transform(label)
# Make_Price_train_test()

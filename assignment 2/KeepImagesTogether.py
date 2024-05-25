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
        # want to keep: Positive, Mostly Positive, Mixed, Negative (if someone wants to play the game, this is )
        mapping = {'Very Positive' : 'Positive', 'Overwhelmingly Positive': 'Positive',
                   'Mostly Negative': 'Negative', 'Overwhelmingly Negative': 'Negative', 'Very Negative' : 'Negative'}
        df.loc[:,'sentiment'] = df['sentiment'].replace(mapping)
        
        # make own labelencoder
        # Positive: 4577, Mixed: 1144, Mostly Positve: 1031, Negative: 152
        le = {'Mixed': 1, 'Mostly Positive': 2, 'Negative': 3, 'Positive': 0}
        df.loc[:,'sentiment'] = df['sentiment'].replace(le)
        # Stratified sampling
        self.train, self.test = train_test_split(df, test_size=0.2, random_state=123, stratify=df[['sentiment']])
        # training_data = df.sample(frac = 0.7, random_state=123)
        # test_data = df.drop(training_data.index)
        self.train = self.train.sample(frac = 1)
        # self.train_sent_im = self.sent_im(train).sample(frac = 1)
        # self.test_sent_im = self.sent_im(test)

        # self.label_encoder = LabelEncoder()
        # df['sentiment'] = self.label_encoder.fit_transform(df['sentiment'])
        # self.training_data = df.sample(frac = 0.7, random_state=123)
        # self.test_data = df.drop(self.training_data.index)

    def get_training_set(self):
        return self.train
    
    def get_test_set(self):
        return self.test
    
    def get_original_label(self, label):
        return self.label_encoder.inverse_transform(label)
# Make_Price_train_test()

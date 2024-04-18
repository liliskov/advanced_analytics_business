from sklearn.model_selection import  train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval
import os

class Make_Sentiment_train_test:
    def __init__(self):
        df = pd.read_csv("dataset.csv")
        df = df[['sentiment', 'screenshots']]
        df = df.dropna(subset = 'sentiment')
        self.label_encoder = LabelEncoder()
        df['sentiment'] = self.label_encoder.fit_transform(df['sentiment'])
        training_data = df.sample(frac = 0.7, random_state=123)
        test_data = df.drop(training_data.index)
        self.train_sent_im = self.sent_im(training_data.copy()).sample(frac = 1)
        self.test_sent_im = self.sent_im(test_data.copy())

    def sent_im(self,data):
        sent_im = data[['sentiment','screenshots']]
        sent_im.loc[:, 'screenshots'] = sent_im['screenshots'].apply(lambda row: literal_eval(row))
        sent_im = sent_im.explode('screenshots')
        sent_im.loc[:, 'screenshots'] = sent_im['screenshots'].apply(lambda x: x.replace('jpg', 'webp'))
        sent_im = sent_im[sent_im['screenshots'].apply(lambda x: os.path.exists(os.path.join('images/', x)))]
        sent_im.to_csv('prices.csv')
        return sent_im

    def get_training_set(self):
        return self.train_sent_im
    
    def get_test_set(self):
        return self.test_sent_im
    
    def get_original_label(self, label):
        return self.label_encoder.inverse_transform(label)
#Make_Sentiment_train_test()
from sklearn.model_selection import  train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval
import os

class Make_Sentiment_train_test_grouped_inference:
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
        train, test = train_test_split(df, test_size=0.2, random_state=123, stratify=df[['sentiment']])
        self.train_sent_im = self.sent_im(train).sample(frac = 1)
        self.train_sent_im= self.train_sent_im[['sentiment', 'screenshots']]
        self.test_sent_im = self.sent_im(test)

    def sent_im(self,data):
        sent_im = data[['sentiment','screenshots']]
        sent_im['id'] = sent_im.index
        sent_im.loc[:,'screenshots'] = sent_im['screenshots'].apply(lambda row: literal_eval(row))
        sent_im = sent_im.explode('screenshots')
        sent_im.loc[:,'screenshots'] = sent_im['screenshots'].apply(lambda x: x.replace('jpg', 'webp'))
        sent_im = sent_im[sent_im['screenshots'].apply(lambda x: os.path.exists(os.path.join('images/', x)))]
        return sent_im

    def get_training_set(self):
        return self.train_sent_im
    
    def get_test_set(self):
        return self.test_sent_im
    
    def get_original_label(self, label):
        return self.label_encoder.inverse_transform(label)
        
        
Make_Sentiment_train_test_grouped_inference()
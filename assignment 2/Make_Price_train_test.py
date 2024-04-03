from sklearn.model_selection import  train_test_split
import pandas as pd
from ast import literal_eval
import os

class Make_Price_train_test:
    def __init__(self):
        df = pd.read_csv("dataset.csv")
        training_data = df.sample(frac = 0.7, random_state=123)
        test_data = df.drop(training_data.index)
        self.train_price_im = self.price_im(training_data.copy())
        self.test_price_im = self.price_im(test_data.copy())

    def price_im(self,data):
        price_im = data[['price','screenshots']]
        price_im.loc[:, 'screenshots'] = price_im['screenshots'].apply(lambda row: literal_eval(row))
        price_im = price_im.explode('screenshots')
        price_im.loc[:, 'screenshots'] = price_im['screenshots'].apply(lambda x: x.replace('jpg', 'webp'))
        #delete all the screenshotrows that do not really have an image
        price_im = price_im[price_im['screenshots'].apply(lambda x: os.path.exists(os.path.join('images/', x)))]
        return price_im
    
    def get_training_set(self):
        return self.train_price_im
    
    def get_test_set(self):
        return self.test_price_im
    

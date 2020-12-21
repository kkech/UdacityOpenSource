
PROCESSED_DATA = "./processed-data"

import torch
import pandas as pd
import numpy as np
import util.model as model
import util.data_prep as data_prep

class Predict(object):
    def __init__(self):
        self.model = model.SimpleLSTM()
        checkpoint = torch.load('data/saved/lstm_model.pt')
        state_dict = checkpoint['net']
        self.model.load_state_dict(state_dict) 
        self.dp = data_prep.Downloader()
        self.getLatestSequence()

    def getLatestSequence(self):
        dataset, year, month, day = self.dp.getLatestData(station = "47267", length = 480, save = False)
        self.year = year
        self.month = month
        self.day = day
        # dataset.set_index(dataset.columns[0], inplace=True)
        self.input_data = np.array(dataset)
        self.mean = np.mean(self.input_data, axis=0)
        self.std = np.std(self.input_data, axis=0) 
        self.input_data = (self.input_data - self.mean)/self.std  
        self.input_data = torch.Tensor(self.input_data).unsqueeze(0)
        self.preds = []
        for count in range(0,24):
            pred = self.model(self.input_data[:, count:480+count, :])
            pred = pred.item() * self.std[0] + self.mean[0] # de-normalization
            self.preds.append(pred)

    def get(self):
        year,month,day, _, _ = self.dp.getTimeAtStation()
        if not ( year is self.year and month is self.month and day is self.day ):
            self.getLatestSequence()
        else:
            print("don't need to update")

        # count = 0
        # for pred in preds:
        #    print("{:02d} - {:2.2f}".format(count, pred))
        #    count = count+1
        
        return self.preds, self.year, self.month, self.day
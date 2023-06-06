import pandas as pd
import torch
from Settings import *
from HelperFunctions import loadReads, fullNormalise
import time
from sklearn.preprocessing import MinMaxScaler




#with sequence length of 4 and 5 sequences of length 10, the "length" of the whole dataset = 5*(10-4) = 30
#indexable entries are concatenation of all squiggles, each truncated by the sequence length
    

class SquigglesDataset(torch.utils.data.Dataset):
    def __init__(self, filepath="../../similar_testdata/similar_squiggles.fast5"):
        self.filepath = filepath
        self.squiggles, self.ids = loadReads()
        print(type(self.squiggles[0]))
        self.squiggles = [fullNormalise(i) for i in self.squiggles]
        print(type(self.squiggles[0]))

        print(self.ids)
        self.startIndices = self.getStarts()
        print(f"no of squiggles: {len(self.startIndices)}")
        #print(self.startIndices)
        self.length = self.startIndices[-1] + len(self.squiggles[-1]) - sequenceLength
        print(f"total length: {self.length}")
        self.scaler = MinMaxScaler(feature_range=(-1,1))         #scales all values between -1 and 1 (the output range of LSTM model)
        

    def getStarts(self):
        starts = [len(i)-sequenceLength for i in self.squiggles]
        starts.insert(0, 0)
        starts.pop(-1)
        for i in range(1, len(starts)):
            starts[i] = starts[i] + starts[i-1]

        return starts


    def __len__(self):
        return self.length


    #TODO: binary search optimisation, also convert to tensor just after reading, also scaling at this stage.
    def __getitem__(self, index):
        #search through startIndices list and get index of value that is below given index
        #then the return item starts at index-startIndex index of that row.
        row = None
        for i in range(len(self.startIndices)):
            if index < self.startIndices[i]:
                row = i-1
                subIndex = index - self.startIndices[i-1]                 #subindex is the index of the number within the squiggle
                break

        if row == None:
            row = len(self.startIndices)-1
            subIndex = index - self.startIndices[-1]
        #print(row)


        x = self.squiggles[row][subIndex:subIndex+sequenceLength]
        y = self.squiggles[row][subIndex+1:subIndex+sequenceLength+1]

        x = self.scaler.fit_transform(x.reshape(-1, 1))
        y = self.scaler.transform(y.reshape(-1, 1))

        x = torch.tensor(x).squeeze(-1)
        y = torch.tensor(y).squeeze(-1)

        
        return (
            x, y
            #torch.tensor(self.squiggles[row][subIndex:subIndex+sequenceLength]),
            #torch.tensor(self.squiggles[row][subIndex+1:subIndex+sequenceLength+1])
        )
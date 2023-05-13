import torch
import torch.nn as nn

#dataset
sequenceLength = 200#60
datapath = "../../similar_testdata/similar_squiggles.fast5"
subset = ["00014eb4-4e2c-4087-ac24-57dea735b7b4", "601f7bc7-3c09-4a78-a9b7-097bddcde809",
          "1b6939ba-4a35-4696-bc6f-2ddb3368266e", "95c03c50-57a5-4092-9a5f-87182d06a12c",
          "b6d8845b-3eec-42f6-9510-b8ba1fc1ec44"]

'''
1 - 00014eb4-4e2c-4087-ac24-57dea735b7b4
2 - 601f7bc7-3c09-4a78-a9b7-097bddcde809
3 - 1b6939ba-4a35-4696-bc6f-2ddb3368266e
4 - 95c03c50-57a5-4092-9a5f-87182d06a12c
5 - b6d8845b-3eec-42f6-9510-b8ba1fc1ec44


1 <--> 2    90.1
3 <--> 4    91.2
4 <--> 5    95.9 or 90.1
'''

#model hyperparameters
input_size = 1
hidden_size = 30
num_layers = 1
dropout = 0
model_path = 'model_weights.pt'


#training
batch_size = 256
epochs = 15
lossFunction = nn.MSELoss()#nn.CrossEntropyLoss()
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#producing embeddings
nGramStep = 20






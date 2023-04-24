import torch
import torch.nn as nn

#dataset
sequenceLength = 30
datapath = "../../similar_testdata/similar_squiggles.fast5"

#model hyperparameters
input_size = 1
hidden_size = 200
num_layers = 1
dropout = 0
model_path = 'model_weights.pt'


#training
batch_size = 256
epochs = 12
lossFunction = nn.MSELoss()#nn.CrossEntropyLoss()
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



import torch
import torch.nn as nn

#dataset
sequenceLength = 200#60
datapath = "../../similar_testdata/similar_squiggles.fast5"
subset = ["9ee3897c-2329-4fc6-9ef3-d8eb97cd4919", "29fd6fb8-0f2f-4aa2-aa1a-c367f327acb6",
          "f5e994e9-38a1-4079-adb5-bdacf43e0a17", "83e5572b-b883-4c88-a500-777144d48134",
          "b61cb37c-ea80-4011-ac7a-6c853fd73eb6", "68b93e56-74fa-431b-9f40-a214a8bfc0b8",
          "93ab5242-dab6-4609-b82c-04746416b6f1", "8edd07ab-777f-4784-ab0b-c268772d3113",
          "edd987c6-0a3f-4214-a630-a1883b4f0f1a", "7f7dfe85-7276-4a9d-ad84-2fa6dfa67635"]

'''
subset = ["00014eb4-4e2c-4087-ac24-57dea735b7b4", "601f7bc7-3c09-4a78-a9b7-097bddcde809",
          "1b6939ba-4a35-4696-bc6f-2ddb3368266e", "95c03c50-57a5-4092-9a5f-87182d06a12c",
          "b6d8845b-3eec-42f6-9510-b8ba1fc1ec44"]
'''
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






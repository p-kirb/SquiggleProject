from torch import optim
from LSTMAutoencoder import LSTM
from SquigglesDataset import SquigglesDataset
from torch.utils.data import DataLoader
from Settings import *
import time
import matplotlib.pyplot as plt



def train(dataloader, model):
    print("Begin training...")
    startTime = time.time()
    model.train()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochLosses = []
    for epoch in range(epochs):
        batchLosses = []
        epochStartTime = time.time()
        state_h, state_c = model.init_state(sequenceLength)

        for batch, (x, y) in enumerate(dataloader):             #updates weights after batch
            

            #TODO: normalise/scale inputs and/or outputs

            x = x.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            #y = y.long()

            optimizer.zero_grad()                               #sets all tensors gradients to 0

            y_pred, (state_h, state_c) = model(x.to(device), (state_h, state_c))


            loss = lossFunction(torch.squeeze(y_pred, -1), y.to(device))
            #if batch % batch_size == 0:
            #    print(f"prediction: {torch.squeeze(y_pred, -1)}")
            #    print(f"target: {y}")

            state_h = state_h.detach()                  #freezing h and c tensors so that gradient descent doesn't update their values.
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
            batchLosses.append(loss.item())

        print(f"Epoch took {time.time()-epochStartTime}s.")
        epochLosses.append(sum(batchLosses)/len(batchLosses))

    
    print(f"Total training took {time.time()-startTime}s.")
    plt.plot(epochLosses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("loss.png")
    return model


dataset = SquigglesDataset(datapath)
dataloader = DataLoader(dataset, batch_size)
model = LSTM()
model = train(dataloader, model)
# save the state dict to a file
torch.save(model.state_dict(), model_path)


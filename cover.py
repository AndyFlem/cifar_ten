
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer as timer

class FaceLandmarksDataset(Dataset):
    """Land cover dataset."""

    def __init__(self, csv_file, train=True):
        if train:
            self.features_frame = pd.read_csv(csv_file,skipfooter=81000,skiprows=000000)
        else: 
            self.features_frame = pd.read_csv(csv_file,skiprows=500000)


    def __len__(self):
        return len(self.features_frame)

    def __getitem__(self, idx):
        feature = self.features_frame.iloc[idx, :].as_matrix().astype('float')
        return feature

if __name__ == '__main__':

    start=timer()

    samples=500
    features,neurons,classes=8,500,7
    learning_rate=1e-6
    sum_loss=0
    epochs=3

    ds=FaceLandmarksDataset('covtype2.csv',train=True)
    dataloader = DataLoader(ds, batch_size=samples,
                            shuffle=True, num_workers=4)

    model=torch.nn.Sequential(
        torch.nn.Linear(features,neurons),
        torch.nn.ReLU(),
        #torch.nn.Linear(neurons,neurons),
        #torch.nn.ReLU(),
        #torch.nn.Linear(neurons,neurons),
        #torch.nn.ReLU(),
        torch.nn.Linear(neurons,classes)
        #torch.nn.LogSoftmax()
    )

    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.NLLLoss()

    for e in range(epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            feats=sample_batched[:,:-1].float()
            classes=sample_batched[:,8].view(samples).long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred=model(feats)
            loss = criterion(y_pred, classes)

            loss.backward()
            sum_loss+=loss.item()
            optimizer.step()

            if i_batch % 100 == 0:
                print("epoch: {:d},batch: {:d}, av loss: {:06.3f}".format(e+1,i_batch,sum_loss/100))
                sum_loss=0

    training_time=timer()


    ds_test=FaceLandmarksDataset('covtype2.csv',train=False)
    dataloader_test = DataLoader(ds_test)

    #w=list(model.parameters())
    #print(w)
    correct_count=0
    for i_test, sample_batched in enumerate(dataloader_test):
        feats=sample_batched[:,:-1].float()
        classes=sample_batched[:,8].view(1)

        y_pred=model(feats)

        _,pred= torch.max(y_pred, 1)

        pred=int(pred.item())
        clss=int(classes.item())

        if clss==pred:
            correct_count+=1


        if (i_test+1) % 2000 == 0:
            print("Tests: {:d}, Correct: {:d}%".format(i_test+1,int(correct_count/(i_test+1)*100)))
        #print("Correct: {}, Class: {:d}, Prediction: {:d}".format(clss==pred,clss,pred))

    end_time=timer()

    print("Training in: {:d} seconds. Accuracy: {:d}%".format(int(training_time-start),int(correct_count/(i_test+1)*100)))
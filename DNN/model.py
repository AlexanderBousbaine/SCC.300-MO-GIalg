import torch
from torch import nn
from torch.utils.data import Dataset

class QueryDataset(Dataset):
    
    def __init__(self, giData):
        self.giDataFrame = giData  
    
    def __len__(self):
        return len(self.giDataFrame)
    
    def __getitem__(self, index):
        fullData = list(self.giDataFrame.iloc[index])
        # Return as pytorch tensor
        return torch.tensor(fullData)



class ResultDataset(Dataset):

    def __init__(self, giData):
        self.giDataFrame = giData  
    
    def __len__(self):
        return len(self.giDataFrame)
    
    def __getitem__(self, index):
        fullData = list(self.giDataFrame.iloc[index])
        # Take label off the end of the row and return it as a separate value
        labels = fullData.pop()
        # Return as pytorch tensors
        return torch.tensor(fullData), torch.tensor(labels)



class MultiLayerPerceptron(nn.Module):
    #Multilayer Perceptron for regression.
    # More layers =ed longer time with no noticeable improvement
    # 2x input nodes gave less variation across predictions than 3x
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(23, 46),
        nn.ReLU(),
        nn.Linear(46, 23),
        nn.ReLU(),
        nn.Linear(23, 1)
    )

    def forward(self, x):
        #Forward pass
        return self.layers(x)
        
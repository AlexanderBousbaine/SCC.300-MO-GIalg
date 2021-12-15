import torch
import glob
import csv
import pandas
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from csv import reader


class GIResultDataset(Dataset):

    def __init__(self, giData):
        self.giDataFrame = giData  
    
    def __len__(self):
        return len(self.giDataFrame)
    
    def __getitem__(self, index):
        fullData = list(self.giDataFrame.loc[index])
        # Take label off the end of the row and return it as a separate value
        labels = fullData.pop()
        # Return as pytorch tensors
        return torch.tensor(fullData), torch.tensor(labels)



class MultiLayerPerceptron(nn.Module):
    #Multilayer Perceptron for regression.
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
        #nn.Linear(23, 64),
        nn.Linear(18, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )


    def forward(self, x):
        #Forward pass
        return self.layers(x)
        
        
# To normalise the 'elite' variables who have a max value of 15 and min of 0
def normaliseElite(num):
    return num/15
    

# To normalise the 'limit' variables who have a max value of 15 and min of 1    
def normaliseLimit(num):
    return (num - 1)/14

if __name__ == '__main__':

    dataPath = "C:/Users/atbou/Documents/Alexander's Work/University Stuff/Modules/SCC.300 - 3rd Year Project/Code/SCC.300-MO-GIalg/DNN/data/*"

    print("in model script")
    
    print("setting up dataset")
    filePaths = glob.glob(dataPath)
    numFiles = len(filePaths)
    # DataFrame definition for new data
    # dataArray = pandas.DataFrame(columns = ["mp", "hgt", "elite", "limit", "mw1", "mw2", "mw3", "mw4", "ow1", "ow2", "ow3", "ow4", "ow5", "ow6", "ow7", "ow8", "ow9", "ow10", "ow11", "ow12", "ow13", "ow14", "ow15", "fitness"]) 
    dataArray = pandas.DataFrame(columns = ["mp", "hgt", "elite", "limit", "mw1", "mw2", "mw3", "mw4", "ow1", "ow2", "ow3", "ow4", "ow5", "ow6", "ow7", "ow8", "ow9", "ow10", "fitness"]) 
    print("got "+str(numFiles)+" files")
    
    # Go through all files in the data folder
    for fileName in filePaths:
        dataFile = open(fileName,  'r')
        csvReader = reader(dataFile)

        # Go through each row in the csv file
        for row in csvReader:
            # Convert each item in the row to a floating point value
            row = [float(i) for i in row]
            # Assing row to the last row in the dataframe
            dataArray.loc[len(dataArray)] = row

        dataFile.close()

    # Normalise 'elite' AND 'limit'
    dataArray['elite'] = dataArray['elite'].apply(normaliseElite)
    dataArray['limit'] = dataArray['limit'].apply(normaliseLimit)
        
    # Create Dataset and DataLoader instances
    # TODO: Carve up into fifths and do the shuffle thing (1 fifth testing, 4 fifths training)
    resultDataset = GIResultDataset(dataArray)
    trainingSet = DataLoader(resultDataset, batch_size=4, shuffle=True, num_workers=1)
    
    model = MultiLayerPerceptron()
    #move model onto GPU now if need be
    
    # Mix of MSE (MSELoss) - which is sensitive to outliers - and MAE (L1Loss) - which works best with lots of outliers
    # May change once I can see how the data looks
    lossFunc = nn.SmoothL1Loss()
    
    # Adam said to be most common optimisation algorithm - so why deviate from the norm?
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Training loop
    for epoch in range(0, 25):
    
        print("Epoch: "+str(epoch))
        
        # To keep track of loss
        epochLoss = 0.0
    
        for batchNo, data in enumerate(trainingSet):
            
            inputData, labels = data
            # Make double sure all values are floats
            inputData, labels = inputData.float(), labels.float()
            # Reshape labels tensor to match the shape of the model output
            labels = labels.reshape((labels.shape[0], 1))
            
            #'''
            # Reset gradients
            optimiser.zero_grad()
            
            # Layers from forward pass
            outputData = model(inputData)
            
            # Get loss
            loss = lossFunc(outputData, labels)
            
            # Perform backward pass
            loss.backward()

            # Optimise weights
            optimiser.step()
            
            epochLoss += loss.item()
            #'''
            
                    
        print('Sum loss of epoch %d: %f' %(epoch + 1, epochLoss))
            
    print("Training Finished")




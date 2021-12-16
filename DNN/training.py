import torch
import model
import glob
import csv
import pandas
from csv import reader
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# To normalise the 'elite' variables which have a max value of 15 and min of 0
def normaliseElite(num):
    return num/15
    

# To normalise the 'limit' variables which have a max value of 15 and min of 1    
def normaliseLimit(num):
    return (num - 1)/14


def analysePredictions(dfRow):
    values = list(dfRow)
    fit = values.pop(0)
    
    avgPred = sum(values)/len(values)
    minPred = min(values)
    maxPred = max(values)
    
    valRange = max(values) - min(values)
    dev = [abs(val - fit) for val in values]
    avgDev = sum(dev)/len(dev)
    
    tendencies = ["high" if val > fit else "low" for val in values]
    tendency = max(set(tendencies), key=tendencies.count)
    
    metaEval.loc[len(metaEval)] = [fit, avgPred, minPred, maxPred, valRange, min(dev), max(dev), avgDev, tendency]
    
    '''
    print(f"Label: {round(fit, 4)}")
    print(f"Predictions {values}")
    print(f"Range of predictions: {round(valRange, 4)}")
    print(f"Average deviation from label: {round(avgDev, 4)}\n")
    '''


if __name__ == '__main__':

    train = False
    loadModel = False
    evaluate = True

    dataPath = "./data/*"

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

    #TODO: Normalise 'fitness' to the range 0-1, discard values > 2

    # Normalise 'elite' AND 'limit'
    dataArray['elite'] = dataArray['elite'].apply(normaliseElite)
    dataArray['limit'] = dataArray['limit'].apply(normaliseLimit)
        
    # Create Dataset and DataLoader instances
    # TODO: Carve up into fifths and do the shuffle thing (1 fifth testing, 4 fifths training)
    resultDataset = model.ResultDataset(dataArray)
    setSize = len(resultDataset)
    
    trainingSet, testingSet = torch.utils.data.random_split(resultDataset, [int((4*setSize/5)), int(setSize/5)])
    
    trainingLoader = DataLoader(trainingSet, batch_size=4, shuffle=True, num_workers=1)
    evaluationLoader = DataLoader(testingSet, batch_size=1, shuffle=True, num_workers=1)
    
    mlp = model.MultiLayerPerceptron()
    #move model onto GPU now if need be
    
    # Mix of MSE (MSELoss) - which is sensitive to outliers - and MAE (L1Loss) - which works best with lots of outliers
    # May change once I can see how the data looks
    lossFunc = nn.SmoothL1Loss()
    
    # Adam said to be most common optimisation algorithm - so why deviate from the norm?
    optimiser = torch.optim.Adam(mlp.parameters(), lr=0.0001)
    
    if(train):
        print("Training")
        print(f"Size of training set: {len(trainingSet)}")
        mlp.train()
        
        if(loadModel):
            print("Loading model and optimiser states")
            mlp.load_state_dict(torch.load("./TrainedModel.pt"))
            optimiser.load_state_dict(torch.load("./ModelOptimiser.pt"))
        
        # Training loop
        for epoch in range(0, 25):
        
            print("Epoch: "+str(epoch))
            
            # To keep track of loss
            epochLoss = 0.0
        
            for batchNo, data in enumerate(trainingLoader):
                
                inputData, labels = data
                # Make doubly sure all values are 32 bit float tensors
                inputData, labels = inputData.float(), labels.float()
                # Reshape labels tensor to match the shape of the model output
                labels = labels.reshape((labels.shape[0], 1))
                
                #'''
                # Reset gradients
                optimiser.zero_grad()
                
                # Get prediction from current model
                outputData = mlp(inputData)
                
                # Get loss from prediction
                loss = lossFunc(outputData, labels)
                
                # Perform backward pass
                loss.backward()

                # Optimise weights
                optimiser.step()
                
                epochLoss += loss.item()
                #'''
                
                        
            print('Sum loss of epoch %d: %f' %(epoch + 1, epochLoss))
        
        print("Training Finished")    
        torch.save(mlp.state_dict(), "./TrainedModel.pt")
        torch.save(optimiser.state_dict(), "./ModelOptimiser.pt")
    
    #Do n (5) training loops over the same data - check the variation in fitness predictions across the loops
    if(evaluate):
        print("Evaluation")
        print(f"Size of testing set: {len(testingSet)}")
        
        predictionFrame = pandas.DataFrame(columns = ["label", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20"])
        metaEval = pandas.DataFrame(columns = ["label", "avgPrediction", "minPrediction", "maxPrediction", "rangeOfPred", "min_devFromLabel", "max_dFL", "avg_dFL", "tendency"])
        
        with torch.no_grad():
            
            if(not train):
                print("Load model before evaluating")
                mlp.load_state_dict(torch.load("./TrainedModel.pt"))
                
            mlp.eval()    
            
            for col in ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20"]:
                print(f"Prediction round: {col}")
                for batchNo, data in enumerate(evaluationLoader):
                    
                    inputs, labels = data
                    # Double check float tensor
                    inputs, labels = inputs.float(), labels.float()
                    labels = labels.reshape((labels.shape[0], 1))

                    # Get prediction
                    prediction = mlp(inputs)
                    
                    predFloat = round(prediction.item(), 3)
                    actFloat = round(labels.item(), 3)
                    
                    # print(f"Pred: {predFloat}\nActual: {actFloat}")
                    
                    predictionFrame.at[batchNo, "label"] = labels.item()
                    predictionFrame.at[batchNo, col] = prediction.item()
                    
            # print(predictionFrame)
            predictionFrame.apply(analysePredictions, axis=1)
            
            print(metaEval)
            
    
    #make sure to do 'model.eval()' before predicting




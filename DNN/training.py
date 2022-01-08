import torch
import model
import glob
import csv
import pandas
import statistics
from sklearn.model_selection import KFold
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
    
    avgPred = statistics.mean(values)
    medPred = statistics.median_high(values)
    minPred = min(values)
    maxPred = max(values)
    
    valRange = max(values) - min(values)
    dev = [abs(val - fit) for val in values]
    avgDev = sum(dev)/len(dev)
    
    tendencies = ["high" if val > fit else "low" for val in values]
    tendency = max(set(tendencies), key=tendencies.count)
    
    pLow = (minPred - fit)
    pUp = (maxPred - fit)
    
    metaEval.loc[len(metaEval)] = [fit, avgPred, medPred, str(round(minPred,6))+" ("+str(round((100*pLow),2))+"%)", str(round(maxPred,6))+", ("+str(round((100*pUp),2))+"%)", valRange, min(dev), max(dev), avgDev, tendency]
    
    '''
    print(f"Label: {round(fit, 4)}")
    print(f"Predictions {values}")
    print(f"Range of predictions: {round(valRange, 4)}")
    print(f"Average deviation from label: {round(avgDev, 4)}\n")
    '''

#TODO: Command line arguments.
#      Try using SKLearn LinearRegression.
#      Try making distribution of input fitnesses more even.
if __name__ == '__main__':

    train = False
    loadModel = True
    evaluate = True
    crossValidate = False
    crossTrain = False
    
    epochs = 30
    folds = 2
    
    if(crossValidate):
        train = True
        evaluate = True
        loadModel = False
        crossTrain = False
        
    if(crossTrain):
        train = True
        evaluate = False

    dataPath = "../Results/*/*.csv"

    print("in model script")
    
    print("setting up dataset")
    filePaths = glob.glob(dataPath)
    numFiles = len(filePaths)
    # DataFrame definition for new data
    dataArray = pandas.DataFrame(columns = ["mp", "hgt", "elite", "limit", "mw1", "mw2", "mw3", "mw4", "ow1", "ow2", "ow3", "ow4", "ow5", "ow6", "ow7", "ow8", "ow9", "ow10", "ow11", "ow12", "ow13", "ow14", "ow15", "fitness"]) 
    
    #Old data dataframe
    # dataArray = pandas.DataFrame(columns = ["mp", "hgt", "elite", "limit", "mw1", "mw2", "mw3", "mw4", "ow1", "ow2", "ow3", "ow4", "ow5", "ow6", "ow7", "ow8", "ow9", "ow10", "fitness"]) 
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
    
    # Remove all rows where fitness values are greater than 1
    bef = dataArray.size
    dataArray = dataArray[dataArray.fitness <= 1.0];
    aft = dataArray.size
    print(f"Removed {bef - aft} rows")
    
    # Create Dataset and DataLoader instances
    resultDataset = model.ResultDataset(dataArray)
        
    tLoaders = []
    eLoaders = []  
    allEvals = []
    losses = []
    
    if(crossValidate or crossTrain):
        # 10 be good number apparently
        folds = 10
        print(f"Performing {folds}-fold cross validation")
    
        
    kf = KFold(n_splits = (10 if folds < 10 else folds))

    for trainIdcs, evalIdcs in kf.split(resultDataset):
        # Create dataloaders for the indicies.
        #print(f"Taining Idices {trainIdcs}")
        #print(f"Evaluation Indicies {evalIdcs}")
        
        trainingSet = model.ResultDataset(dataArray.iloc[trainIdcs])
        evaluationSet = model.ResultDataset(dataArray.iloc[evalIdcs])
        
        '''
        print(f"Size of total set: {len(resultDataset)}")
        print(f"Size of training set: {len(trainingSet)}")
        print(f"Size of evaluation set: {len(evaluationSet)}")   
        '''
        
        tLoaders.append(DataLoader(trainingSet, batch_size=4, shuffle=True, num_workers=1))
        eLoaders.append(DataLoader(evaluationSet, batch_size=4, shuffle=True, num_workers=1))
        
    if(folds == 2):
        folds = 1 
        
    for f in range(folds):
        print(f"Starting fold: {f+1}")
        # Load in appropriate dataloaders
        trainingLoader = tLoaders[f]
        evaluationLoader = eLoaders[f]
        
        # In cross-validation, model is discarded and remade with every interation
        if(f == 0 or crossValidate):
            print("Init Model")
            # Model resets with every fold
            mlp = model.MultiLayerPerceptron()
            #move model onto GPU now if need be
        
            # SmoothL1Loss is a mix of MSE (MSELoss) - which is sensitive to outliers - and MAE (L1Loss) - which works best with lots of outliers
            # May change once I can see how the data looks
            lossFunc = nn.SmoothL1Loss()
        
            # Adam said to be most common optimisation algorithm - haha sheep go baaa
            optimiser = torch.optim.Adam(mlp.parameters(), lr=0.001)
            
            #Scheduler to change the learning rate of the optimiser every n epochs
            scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size = epochs, gamma = 0.1, verbose = False)
    
        if(train):
            print("Training")
            mlp.train()
            
            if(loadModel):
                print("Loading model and optimiser states")
                mlp.load_state_dict(torch.load("./TrainedModel.pt"))
                optimiser.load_state_dict(torch.load("./ModelOptimiser.pt"))
            
            # Training loop
            for epoch in range(epochs):
            
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
                    
                            
                print(f"Sum loss of epoch {epoch}: {round(epochLoss,5)}")
                print(f"Average loss {round((epochLoss/len(trainingLoader)), 6)}")
                
                scheduler.step()
            
            losses.append(epochLoss)
            print("Training Finished")    
            torch.save(mlp.state_dict(), "./TrainedModel.pt")
            torch.save(optimiser.state_dict(), "./ModelOptimiser.pt")
            print("Model Saved")
        
        # Does 20 prediction loops over the same data - checks the variation in fitness predictions across the loops
        # Increase batch size
        if(evaluate):
            print("Evaluation")
            print(f"Size of testing set: {len(evaluationSet)}")
            
            predictionFrame = pandas.DataFrame(columns = ["label", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20"])
            metaEval = pandas.DataFrame(columns = ["label", "avgPred", "medPred", "minPred", "maxPred", "rangeOfPred", "min_devFromLabel", "max_dFL", "avg_dFL", "tendency"])
            
            with torch.no_grad():
                
                # If model has not just been trained, load in a model
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
                        
                        counter = 0
                        for i in range(prediction.size(dim=0)):
                            predictionFrame.at[(4*batchNo)+i, col] = prediction[i].item()
                            predictionFrame.at[(4*batchNo)+i, "label"] = labels[i].item()  
                        
                # print(predictionFrame)
                predictionFrame.apply(analysePredictions, axis=1)
                
                allEvals.append(metaEval)
                #print(f"Evaluation of fold: {f}")
                #print(metaEval)
                
    
    print("Showing results")
    for i in range(len(allEvals)):
        print(f"Fold {i}")
        print(allEvals[i])
    
    print(losses)
    #make sure to do 'model.eval()' before predicting




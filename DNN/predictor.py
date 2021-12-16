import model
import torch
import glob
import csv
import pandas
from csv import reader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# To normalise the 'elite' variables which have a max value of 15 and min of 0
def normaliseElite(num):
    return num/15
    

# To normalise the 'limit' variables which have a max value of 15 and min of 1    
def normaliseLimit(num):
    return (num - 1)/14

if __name__ == "__main__":

    queryPath = "./queries/*"

    print("in query script")
    
    print("setting up dataset")
    filePaths = glob.glob(queryPath)
    numFiles = len(filePaths)
    # DataFrame definition for new data
    # dataArray = pandas.DataFrame(columns = ["mp", "hgt", "elite", "limit", "mw1", "mw2", "mw3", "mw4", "ow1", "ow2", "ow3", "ow4", "ow5", "ow6", "ow7", "ow8", "ow9", "ow10", "ow11", "ow12", "ow13", "ow14", "ow15", "fitness"]) 
    dataArray = pandas.DataFrame(columns = ["mp", "hgt", "elite", "limit", "mw1", "mw2", "mw3", "mw4", "ow1", "ow2", "ow3", "ow4", "ow5", "ow6", "ow7", "ow8", "ow9", "ow10", "fitness"]) 
    predictionFrame = pandas.DataFrame(columns = ["prediction"]) 
    
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
        
    # Create Dataset and DataLoader instances on query data
    # Use model.QueryDataset for csvs without attached on the end
    # queryDataset = model.QueryDataset(dataArray)
    queryDataset = model.ResultDataset(dataArray)
    queryLoader = DataLoader(queryDataset, batch_size=1, shuffle=False, num_workers=1)

    with torch.no_grad():
        
        mlp = model.MultiLayerPerceptron()
        mlp.load_state_dict(torch.load("./TrainedModel.pt"))
        mlp.eval()
        
        for batchNo, data in enumerate(queryLoader):
            
            inputs, labels = data
            # Double check float tensor
            inputs, labels = inputs.float(), labels.float()

            # Get prediction
            # prediction = model(data)
            prediction = mlp(inputs)
            
            #print(lables.item())
            #print(prediction)
            
            # Save prediction into DataFrame
            predictionFrame.loc[len(predictionFrame)] = prediction.item()
        
        # Add predictions on the end of the data that was fed in
        dataArray = dataArray.assign(prediction=predictionFrame["prediction"])
        print("Finished predicting, writing results")
        dataArray.to_csv("./predictions/predictedFitness.csv")
            
        
        
    
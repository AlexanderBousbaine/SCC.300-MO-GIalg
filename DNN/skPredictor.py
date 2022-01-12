import model
import torch
import glob
import csv
import pandas
from csv import reader
import joblib

# To normalise the 'elite' variables which have a max value of 15 and min of 0
def normaliseElite(num):
    return num/15
    

# To normalise the 'limit' variables which have a max value of 15 and min of 1    
def normaliseLimit(num):
    return (num - 1)/14

if __name__ == "__main__":

    queryPath = "./queries/*.csv"
    modelName = "skModel.pkl"

    print("in query script")
    
    print("setting up dataset")
    filePaths = glob.glob(queryPath)
    numFiles = len(filePaths)
    # DataFrame definition for new data
    dataArray = pandas.DataFrame(columns = ["mp", "hgt", "elite", "limit", "mw1", "mw2", "mw3", "mw4", "ow1", "ow2", "ow3", "ow4", "ow5", "ow6", "ow7", "ow8", "ow9", "ow10", "ow11", "ow12", "ow13", "ow14", "ow15", "fitness"])
    predictionFrame = pandas.DataFrame(columns = ["prediction"]) 
    
    print("got "+str(numFiles)+" files")
    
    # Go through all files in the data folder
    for fileName in filePaths:
        dataFile = open(fileName, 'r')
        csvReader = reader(dataFile)

        # Go through each row in the csv file
        for row in csvReader:
            # Convert each item in the row to a floating point value
            row = [float(i) for i in row]
            # Assing row to the last row in the dataframe
            dataArray.loc[len(dataArray)] = row

        dataFile.close()

    #for testing files, get rid  of fitness column
    if('fitness' in dataArray):
        fitVals = dataArray.pop('fitness')
    
    # Normalise 'elite' AND 'limit'
    dataArray['elite'] = dataArray['elite'].apply(normaliseElite)
    dataArray['limit'] = dataArray['limit'].apply(normaliseLimit)
            
    mlp = joblib.load("./"+modelName)
    
    predictions = mlp.predict(dataArray.to_numpy(dtype="float"))
    predictions = predictions/1000000
    
    predictionFrame['prediction'] = predictions
    
    #Add data back in for comparison's sake
    if('fitVals' in locals()):
        dataArray = dataArray.assign(fitness=fitVals)
    
    # Add predictions on the end of the data that was fed in
    dataArray = dataArray.assign(prediction=predictionFrame["prediction"])
    print("Finished predicting, writing results")
    dataArray.to_csv("./predictions/predictedFitness.csv")
            
        
        
    
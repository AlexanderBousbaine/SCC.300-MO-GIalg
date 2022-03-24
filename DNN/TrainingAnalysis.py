
import model
import glob
import pandas
import statistics as stats
from sklearn.model_selection import KFold
from csv import reader
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == '__main__':
    folds = 2
    allData = True

    dataPath = "../Results/*/*.csv"
    modelName = "skModel.pkl"

    print("Analysing training data")
    
    print("setting up dataset")
    filePaths = glob.glob(dataPath)
    numFiles = len(filePaths)
    # DataFrame definition for new data
    dataArray = pandas.DataFrame(columns = ["mp", "hgt", "elite", "limit", "mw1", "mw2", "mw3", "mw4", "ow1", "ow2", "ow3", "ow4", "ow5", "ow6", "ow7", "ow8", "ow9", "ow10", "ow11", "ow12", "ow13", "ow14", "ow15", "fitness"]) 
    print("got "+str(numFiles)+" files")
    
    # Go through all files in the data folder
    for fileName in filePaths:
        dataFile = open(fileName,  'r')
        csvReader = reader(dataFile)

        # Go through each row in the csv file
        for row in csvReader:
            # Convert each item in the row to a floating point value - check for strange case where empty strings appear
            row = [float(i) if len(i) > 0 else None for i in row]
            # Assing row to the last row in the dataframe
            dataArray.loc[len(dataArray)] = row

        dataFile.close()
    
    # Remove all rows where fitness values are greater than 2
    # Keep some of those greater than 1 as it is possible for that to occur normally if a chromosome just performs really badly.
    bef = len(dataArray)
    dataArray = dataArray[dataArray.fitness <= 2.0]
    aft = len(dataArray)
    print(f"Removed {bef - aft} row(s) for abnormal fitness values")

    print(f"Size of all data: {len(dataArray)}")

    tLoaders = []
    eLoaders = []

    if(allData):
        folds = 2
        tLoaders.append(dataArray)

    else:
        folds = 1

    # Create Dataset and DataLoader instances
    resultDataset = model.ResultDataset(dataArray)
    
    trainingSet = []
    evaluationSet = []
        
    kf = KFold(n_splits = (10 if folds < 10 else folds))

    for trainIdcs, evalIdcs in kf.split(resultDataset):
        # Create dataloaders for the indicies.
        trainingSet = dataArray.iloc[trainIdcs]
        evaluationSet = dataArray.iloc[evalIdcs]
        
        tLoaders.append(trainingSet)
        eLoaders.append(evaluationSet)

    compFrame = pandas.DataFrame(index = [1, 2], columns = ["numConfigs", "uniqueValues", "mp", "hgt", "elite", "limit", "mw1", "mw2", "mw3", "mw4", "ow1", "ow2", "ow3", "ow4", "ow5", "ow6", "ow7", "ow8", "ow9", "ow10", "ow11", "ow12", "ow13", "ow14", "ow15", "fitness"])

    for fold in range(folds):
        # Load in appropriate datasets
        trainingData = tLoaders[fold]

        print(f"Size of set to analyse: {len(trainingData)}")
        compFrame.at[fold+1, "numConfigs"] = len(trainingData)

        # Round fitness data so that it gets aggregated too
        trainingData = trainingData.round({'fitness': 3})
        sampleData = []

        # Plot and process data
        numPlots = len(trainingData.columns)
        numCols = 2
        numRows = int(numPlots/numCols)
        currentPlot = 1
        plots = []

        f = plt.figure(figsize=(12, 6))

        simScore = 0
        for col in trainingData.columns:
            aggregates = Counter(trainingData[col])
            minVal = min(trainingData[col])
            maxVal = max(trainingData[col])
            mean = stats.mean(trainingData[col])
            stdev = stats.stdev(trainingData[col])
            localSimScore = 0

            for data in trainingData[col]:
                if(aggregates.get(data) == 1):
                    simScore+=1
                    localSimScore+=1

            plt.subplot(1, 2, (2 if currentPlot%2==0 else 1))

            x = list(aggregates.keys())
            y = list(aggregates.values())
            plt.scatter(x, y)
            plt.title(col)
            plt.xlabel(f"Min: {minVal}, Max: {maxVal}\nMean: {mean:.4f}, STDev: {stdev:.4f}")

            compFrame.at[fold+1, col] = f"min: {minVal}, max: {maxVal}, mean: {mean:.4f}, stdev: {stdev:.4f}, uniques: {localSimScore} / {len(trainingData)}"

            if(currentPlot%2 == 0):
                plots.append(f)
                f = plt.figure(figsize=(12, 6))

            currentPlot+=1
                
        print(f"Unique Values: {simScore}")
        a = len(trainingData)
        b = len(compFrame.columns)-2
        compFrame.at[fold+1, "uniqueValues"] = f"{simScore} / {a*b}"

        pdf = PdfPages(f"Attribute_Spread_{('all' if fold == 0 and folds > 1 else 'sample')}.pdf")
        for p in plots:
            pdf.savefig(p)

        pdf.close()

        #Close plots to save memory
        plt.close('all')

    compFrame = compFrame.T

    f, ax = plt.subplots()
    f.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = pandas.plotting.table(ax, compFrame, loc='center')
    table.set_fontsize(14)
    f.savefig("Comparison.pdf", format='pdf', bbox_inches='tight')
    
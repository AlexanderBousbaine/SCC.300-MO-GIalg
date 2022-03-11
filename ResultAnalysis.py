import pandas
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_pdf import PdfPages

def load(files):
    li = []
    all_files = glob.glob(files)
    for filename in all_files:
        # print(filename)
        df = pandas.read_csv(filename, header=None)
        df = df.iloc[:, -1]
        li.append(df)
        
    frame = pandas.concat(li, axis=1, ignore_index=True)
    return frame

def plotDataFromFolder(folderName, show = True):
    filesPath = foldersPath + folderName + "/*.csv"
                
    # Loads fitness columns from all files
    # Gives dataframe with 10 columns and 12 rows - each column is a generation
    fitnesses = load(filesPath)
    
    bestFitnesses = pandas.DataFrame(columns=["Generation", "Fitness", "StdErrMean"])
    for v in fitnesses.columns:
        bfit = fitnesses.iloc[:, v].min()
        err = stats.sem(fitnesses.iloc[:, v])
        bestFitnesses.loc[len(bestFitnesses)] = [v, bfit, err]
    
    bestFitnesses.plot(x="Generation", y="Fitness",grid=True,title="Best Fitness from each Generation of the MOalg in "+folderName)
    
    model = LinearRegression()
    xData = np.array(bestFitnesses['Generation'])
    yData = np.array(bestFitnesses['Fitness'])
    errData = list(bestFitnesses['StdErrMean'])
    
    xData = xData.reshape(len(bestFitnesses), 1)
    yData = yData.reshape(len(bestFitnesses), 1)
    
    model.fit(xData, yData)
    bfl = model.predict(xData)
    
    plt.plot(xData, bfl, 'r')

    plt.errorbar(xData, yData, yerr = errData, fmt='_')
    
    plt.show()
    
    #plotData = pandas.DataFrame(columns = ["Generation", "Fitness_Values"])
    
    #for g in range(0, 9):
    #    plotData.loc[len(plotData)] = [g, fitnesses[g]]
    
    #print(plotData)
    
    #fitnesses.plot.scatter(x="Gens", y=9, grid=True)
    #plt.show()

if __name__ == "__main__":

    fileNum = 0
    if(len(sys.argv) == 2):
        fileNum = int(sys.argv[1])

    # Read in results
    foldersPath = "./Results/"
    folders = os.listdir(foldersPath)
    # Go through all folders (Runs of the algorithm) and create graphs of their data
    if(fileNum == 0):
        numFolders = 0
        for folder in folders:
            print(folder)
            if(not folder.endswith(".png") and not folder.endswith(".txt")):
                plotDataFromFolder(folder)

                
    else:
        try:
            plotDataFromFolder(f"Run{fileNum}")
        except:
            print(f"Could not plot the data from folder folder {fileNum}, it might not exist.")

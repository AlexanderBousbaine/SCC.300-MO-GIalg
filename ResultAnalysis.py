from cProfile import label
from turtle import title
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

    plt.figure()
    
    bestFitnesses = pandas.DataFrame(columns=["Generation", "b_Fitness", "m_Fitness", "w_Fitness", "StdErrMean"])
    for v in fitnesses.columns:
        bfit = fitnesses.iloc[:, v].min()
        mfit = sum(fitnesses.iloc[:, v])/len(fitnesses.iloc[:, v])
        wfit = fitnesses.iloc[:, v].max()
        err = stats.sem(fitnesses.iloc[:, v])
        bestFitnesses.loc[len(bestFitnesses)] = [v, bfit, mfit, wfit, err]

        #plot all datapoints
        allY = [i for i in list(fitnesses.iloc[:, v])]
        allX = [v+1 for i in allY]
        plt.plot(allX, allY, 'bo', alpha=0.08)

    model = LinearRegression()
    xData = np.array(bestFitnesses['Generation'])+1
    yData = np.array(bestFitnesses['b_Fitness'])
    yData2 = np.array(bestFitnesses['w_Fitness'])
    yData3 = np.array(bestFitnesses['m_Fitness'])
    errData = list(bestFitnesses['StdErrMean'])
    errData = [[0*x for x in errData], errData]
    
    xData = xData.reshape(len(bestFitnesses), 1)
    yData = yData.reshape(len(bestFitnesses), 1)
    
    model.fit(xData, yData)
    bfl = model.predict(xData)

    #best fitness
    plt.plot(xData, yData, c='b', label="Best Fitness")

    #best fit line
    plt.plot(xData, bfl, 'r', label="Best-fit of Best Fitness")

    #error bars of best fitness
    plt.errorbar(xData, yData, yerr=errData, fmt='_', c="orange", label="Best Fitness StdError")

    #plot worst data
    plt.plot(xData, yData2, c='m', label="Worst Fitness")

    #plot average data
    plt.plot(xData, yData3, c='y', label="Mean Fitness")

    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    plt.grid()
    plt.title(label=f"Fitnesses from each Generation of the MOalg in {folderName}")
    plt.tight_layout()

    if(show):
        plt.show()


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
        plots = []
        for folder in folders:
            print(folder)
            if(not folder.endswith(".png") and not folder.endswith(".txt")):
                plotDataFromFolder(folder)

                
    else:
        try:
            plotDataFromFolder(f"Run{fileNum}")
        except:
            print(f"Could not plot the data from folder folder {fileNum}, it might not exist.")

import pandas
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

def load(files):
    li = []
    all_files = glob.glob(files)
    for filename in all_files:
        # print(filename)
        df = pandas.read_csv(filename, header=None)
        df = df.iloc[:, -1]
        li.append(df)
        
    frame = pandas.concat(li, axis=1, ignore_index=True)
    #frame.columns = ["gen1", "gen2", "gen3", "gen4", "gen5", "gen6", "gen7", "gen8", "gen9", "gen10"]
    return frame

if __name__ == "__main__":
    # Read in results
    foldersPath = "./Results/"
    folders = os.listdir(foldersPath)
    # Go through all folders (Runs of the algorithm) and create graphs of their data
    for folder in folders:
        if(not folder.endswith(".png")):    
            
            filesPath = foldersPath + folder + "/*.csv"
            
            # Loads fitness columns from all files
            # Gives dataframe with 10 columns and 12 rows - each column is a generation
            fitnesses = load(filesPath)
            #fitnesses.insert(0, "Gens", ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12"])
            
            bestFitnesses = pandas.DataFrame(columns=["Generation", "Fitness", "StdErrMean"])
            for v in fitnesses.columns:
                bfit = fitnesses.iloc[:, v].min()
                err = stats.sem(fitnesses.iloc[:, v])
                bestFitnesses.loc[len(bestFitnesses)] = [v, bfit, err]
            
            bestFitnesses.plot(x="Generation", y="Fitness",grid=True,title="Best Fitness from each Generation of the MOalg in "+folder)
            
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
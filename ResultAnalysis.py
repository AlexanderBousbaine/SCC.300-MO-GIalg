import pandas
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

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
            
            bestFitnesses = pandas.DataFrame(columns=["Generation", "Fitness"])
            for v in fitnesses.columns: #[0,1,2,3,4,5,6,7,8,9]:
                bfit = fitnesses.iloc[:, v].min()
                bestFitnesses.loc[len(bestFitnesses)] = [v+1, bfit]
            
            bestFitnesses.plot(x="Generation", y="Fitness",grid=True,title="Best Fitness from each Generation of the MOalg")
            plt.show()
            
            #plotData = pandas.DataFrame(columns = ["Generation", "Fitness_Values"])
            
            #for g in range(0, 9):
            #    plotData.loc[len(plotData)] = [g, fitnesses[g]]
            
            #print(plotData)
            
            #fitnesses.plot.scatter(x="Gens", y=9, grid=True)
            #plt.show()
    
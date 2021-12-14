import pandas as pd
import seaborn as sb
import glob
import math
import matplotlib.pyplot as plt

import sys
import os

# adapted from GIalgorithm graph producing code

def makeCsv(files):
    li = []
    all_files = glob.glob(files)
    for filename in all_files:
        # print(filename)
        pre, ext = os.path.splitext(filename)
        os.rename(filename, pre + ".csv")
        df = pd.read_csv(pre + ".csv")

def load(files):
    li = []
    all_files = glob.glob(files)
    for filename in all_files:
        # print(filename)
        df = pd.read_csv(filename)
        df.columns = [col.strip() for col in df.columns]
        # print(df.columns)
        # get original function values (multiple in some cases)
        original_fitness = df.at[0, "hash_fitness"]
        original_performance_1 = df.at[0, "hash_performance_1"]
        original_performance_0 = df.at[0, "hash_performance_0"]
        # remove original function from frame
        df.drop(0, axis=0)
        # add relative fitness (with group by to match metrics)
        df["rel_fitness"] = df.hash_fitness / original_fitness
        # add relative performance (with group by to match metrics)
        df["rel_performance_1"] = df.hash_performance_1 / original_performance_1
        df["rel_performance_0"] = df.hash_performance_0 / original_performance_0
        # print(df.columns)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame


def best_data(data):
    # print(data.columns)
    idx = data.groupby(['run_name', 'run_number', 'generation'])['hash_fitness'].transform(min) == data['hash_fitness']
    return data[idx]


if __name__ == '__main__':
    # Load data in and add relative fitness and performance columns and remove the id 0 individuals
    
    makeCsv("/home/abousbaine/GIalgorithm/output_files/MOGI/*/phyloLog")
    
    outFolder ="/home/abousbaine/GIalgorithm/output_files/MOGI"
    
    fullData = load(outFolder+"/*/phyloLog.csv")
    
    # Find number of folders in the output folder
    numRuns = len(next(os.walk(outFolder))[1])
    
    # lastGen = 99
    # numRuns = 12

    bestData = best_data(fullData)
    relevantData = bestData[["generation", "run_number", "rel_fitness"]].copy()
    relevantData = relevantData.drop_duplicates()
    relevantData.sort_values(by=["generation", "run_number"], inplace = True)
    relevantData = relevantData.loc[relevantData["run_number"] == numRuns]
    
    # finds the largest value in the 'generation' column and takes that as the largest generation the code reached 
    lastGen = relevantData["generation"].max()
    
    relevantData = relevantData.loc[relevantData["generation"] >= lastGen-5]
    fitnessVals = relevantData["rel_fitness"]
    
    #mean fitness of the last 5 generations from the last run
    avg5Fitness = fitnessVals.mean()
    # endFitness = float(fitnessVals.tail(1))
    
    fFile = open("fitness.txt", "w")
    fFile.write(str(avg5Fitness))
    fFile.close()

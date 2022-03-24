import os
import glob
import pandas

#Load in files
def load(files):
    li = []
    all_files = glob.glob(files)
    for filename in all_files:
        df = pandas.read_csv(filename, header=None)
        df.drop(columns=df.columns[-1], axis=1, inplace=True)
        li.append(df)
        
    frame = pandas.concat(li, axis=0, ignore_index=True)
    return frame

#Count number of uniqye configurations
def countUniqueConfigs(folders):
    uniqueConfigs={}
    totalConfigs = 0
    #Go through each results folder
    for folder in folders:
        if(not folder.endswith(".png") and not folder.endswith(".txt")):
            #load in data as dataframe
            currentFrame = load(foldersPath + folder + "/*.csv")

            # https://stackoverflow.com/questions/37877708/how-to-turn-a-pandas-dataframe-row-into-a-comma-separated-string?answertab=scoredesc#tab-top
            x = currentFrame.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')
            vals = [','.join(ele.split()) for ele in x]
            #
            #Add rows to dictionary using string version of row as key
            for s in vals:
                uniqueConfigs[s] = 1
                totalConfigs += 1

    print(f"{len(uniqueConfigs)} / {totalConfigs} or {(len(uniqueConfigs) / totalConfigs)*100}%")
    return (len(uniqueConfigs), totalConfigs)
            

if(__name__=="__main__"):
    #Total number of options.
    #       mp   hgt   e   mw   mw   mw   mw   ow1  ow2  ow3  ow4  ow5  ow6  ow7  ow8  ow9 ow10 ow11 ow12 ow13 ow14 ow15
    totalPossible = 20 * 20 * 10 * 20 * 20 * 20 * 20 * 20 * 20 * 20 * 20 * 20 * 20 * 20 * 20 * 20 * 20 * 20 * 20 * 20 * 20 * 20

    # Calculate those taken
    foldersPath = "./Results/"
    folders = os.listdir(foldersPath)

    configs, tConfigs = countUniqueConfigs(folders)

    print(f"{configs} / {totalPossible} or {(configs/totalPossible)*100}%")

    
            

    




import numpy as np
"""
<<<<<<< Updated upstream
from Dataloader import dataloader
from MakeDict import findEDF
from MNEplotter import MNEPlotter
=======

from MakeDict import findEDF
from MNEplotter import MNEPlotter
from Dataloader import dataloader
>>>>>>> Stashed changes
"""

def make_batch(IDXlist, size=24, Nlable=6):
    """
    make balance dataset, sampling with replacement.
    """
    filelist = []
    windowlist = []

    if size % Nlable != 0:
        raise Exception(f"Batch size must be devedible by {Nlable}")
    fromset = int(size / Nlable)

    for i in range(Nlable):
        elements = np.random.randint(0, len(IDXlist[i]), fromset)
        for e in elements:
            window = int(IDXlist[i][e][0])
            try:  # See if window already is in list else append it
                winidx = windowlist.index(window)
                filelist[winidx].append(int(IDXlist[i][e][1]))
            except ValueError:
                windowlist.append(window)
                filelist.append([int(IDXlist[i][e][1])])

    return windowlist, filelist
"""

data_path=r"C:\ Users\Andre\Desktop\Deeplearning local\ artifact_dataset\ artifact_dataset"
path=r"artifact_dataset"
idx=[0]
edfDict=findEDF(DataDir=data_path)
#annoIDX=np.load('testIDX.npy',allow_pickle=True)
loader=dataloader(Time_interval=1,Overlap=0.25,Data_paht=data_path)
edfDict,annothlist=loader.anno_mapping(edfDict,index=idx)

#data=loader.loadDict(edfDict=edfDict,index=idx)


windowslist,filelist=make_batch(IDXlist=annothlist,size=1,Nlable=1)

Batch,Batch_X,Batch_Y=loader.loadBatch(edfDict=edfDict,windowlist=windowslist,filelist=filelist)
pl=MNEPlotter(CH_names=loader.CH_names,lableEncoding=loader.one_hot_eoncoding)
pl.plot_raw(Batch_X[0],Batch_Y[0])
"""

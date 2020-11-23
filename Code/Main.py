
import numpy as np

from Dataloader import dataloader
from MakeDict import findEDF


data_path=r"C:\Users\Andre\Desktop\Deeplearning local\artifact_dataset\artifact_dataset"
path=r"artifact_dataset"
idx=np.arange(0,10)
edfDict=findEDF(DataDir=data_path)
loader=dataloader(Time_interval=1,Overlap=0,Data_paht=data_path)
data=loader.loadDict(edfDict=edfDict,index=idx)
print(data)


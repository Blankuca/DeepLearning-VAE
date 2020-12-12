#Qiuqk test of bug appering for certain files.
from MakeDict import findEDF
from Dataloader import dataloader,preprossing

#Part to bad file
data_path=r"C:\Users\Andre\Desktop\Deeplearning local\artifact_dataset\artifact_dataset"

loader=preprossing(Time_interval=1,Overlap=0.75,Data_paht=data_path)
edfDict=findEDF(DataDir=data_path)
loader.auto_dataset(edfDict,"min",file_name="075_overlab_dataset")

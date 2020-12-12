#Qiuqk test of bug appering for certain files.
from MakeDict import findEDF
from Dataloader import dataloader,preprossing

#Part to bad file
data_path=r"C:\Users\Andre\Desktop\Deeplearning local\artifact_dataset\artifact_dataset"

test_file={"path": r"01_tcp_ar\067\00006746\s004_2010_12_06\00006746_s004_t000"}


files={"path": r"\02_tcp_le\010\00001006\s001_2003_04_28\00001006_s001_t001"}
loader=preprossing(Time_interval=1,Overlap=0,Data_paht=data_path)
edfDict=findEDF(DataDir=data_path)
loader.auto_dataset(edfDict,"min",file_name="Zero_overalab_dataset")

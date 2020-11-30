
import numpy as np

import os, mne, re
from datetime import datetime
from mne.io import read_raw_edf
from collections import defaultdict
path=os.path.join(os.getcwd(),r"Data\s007_2012_07_25\00001154_s007_t000")
import pandas as pd



class dataloader:
    def __init__(self,Time_interval,Overlap,Data_paht=None,freq=250,Chanels=None):
        """
        tWindow is how long each window is in seconds
        Overlap given as a float eg: 0.25 for 0.25% overlap
        """
        self.one_hot_eoncoding=np.array(["null","eyem","chew","shiv","elpp","musc"])
        self.freq=freq
        self.tWindow=Time_interval
        self.Overlap=Overlap

        if Chanels==None:
            self.CH_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5','T6','Fz', 'Cz', 'Pz']
        else:
            self.CH_names=Chanels


        if Data_paht is None:
            self.DataDir=os.getcwd()
        else:
            self.DataDir=Data_paht

    def loadDict(self,edfDict,index):
        windowDict = defaultdict()
        for idx in index:
            edfDict[idx] = self.readRawEdf(edfDict[idx])
            windowDict[idx] = self.slidingWindow(edfDict[idx])
        return windowDict

    def loadBatch(self,edfDict,windowlist,filelist):
        batch={}
        X=[]
        Y=[]
        n=0
        for i,idx in enumerate(windowlist):
            edfDict[idx] = self.readRawEdf(edfDict[idx])
            windows=self.slidingWindow(edfDict[idx])
            for w in filelist[i]:
                batch[n]=windows[f"window_{w}"]
                batch[n]['info']={"File index": i, "window index": w}
                X.append(windows[f"window_{w}"]['X'])
                Y.append(windows[f"window_{w}"]['Y'])
                n+=1
        return batch,X,Y


    def loadFile(self,path):
        pass

    def anno_mapping(self,edfDict):
        keys=edfDict.keys()
        annothlist=[[] for i in self.one_hot_eoncoding]
        lablelist=list(self.one_hot_eoncoding)
        for key in keys: #Go over each file.
            anno=self.annoTUH(self.DataDir+"/"+edfDict[key]["path"]+r".tse")
            tN=int(anno.iloc[-1,1]) #Get recording endtine
            t0=int(anno.iloc[0,0]) #Get startime
            file_overwiew=[]
            for i,t in enumerate(range(t0, tN, self.tWindow*(1-self.Overlap))): #Go over the file.
                tEnd=t+self.tWindow
                iStart = sum(anno[0] <= t) - 1
                iEnd = sum(anno[1] <= tEnd)
                if iStart == iEnd:
                    # Assing lable as last started artifact.
                    lable = anno.iloc[iStart, 2]
                else:
                    # if and artifact end in the window assign lable to domenet artifact
                    lable = anno.iloc[[iStart, iEnd][np.argmax([t - anno.iloc[iStart, 0], tEnd - anno.iloc[iStart, 1]])], 2]

                lableidx=lablelist.index(lable)
                annothlist[lableidx].append([key,i])
                file_overwiew.append(lable)
                edfDict[key]["lables"]=file_overwiew






        return edfDict,annothlist

    def readRawEdf(self,edfDict=None,read_raw_edf_param={'preload':True, "stim_channel":"auto"}):
        """
        Function to read the edf file in one given location.
        """
        #EEGseries= read_raw_edf(self.DataDir+"/"+edfDict["path"]+".edf", **read_raw_edf_param)
        EEGseries=read_raw_edf(self.DataDir+"/"+edfDict["path"]+".edf",preload= True)
        #EEGseries.plot()
        EEGseries.resample(sfreq=self.freq)  # Down sample to desired frequense
        EEGseries=self.TUHfooDef(EEGseries)
        EEGseries.pick_channels(self.CH_names)
        EEGseries.set_montage(mne.channels.make_standard_montage(kind="standard_1005", head_size=0.095))
        print(EEGseries.info["ch_names"])
        print(len(EEGseries.info["ch_names"]))
        EEGseries.set_eeg_reference() #Using avage as reference 
        edfDict["rawData"] =EEGseries
        edfDict["Annotation"] = self.annoTUH(self.DataDir+"/"+edfDict["path"]+r".tse")
        #EEGseries.plot()

        # comment this out to get meta data on recording time stamps, WARNING will given and error in python 3.7
        # tStart = edfDict["rawData"].annotations.orig_time-timedelta(hours=1)
        # tLast = int((1+edfDict["rawData"].last_samp)/edfDict["rawData"].info["sfreq"])
        # edfDict["t0"] = tStart
        # edfDict["tN"] = tStart + timedelta(seconds=tLast)

        if self.CH_names != EEGseries.info["ch_names"]:
            raise Exception(f"Montage Error")
        return edfDict

    def TUHfooDef(self,EEGseries=False):
        """
        Gets the chanels reigt
        """
        for i in EEGseries.info["ch_names"]:
            reSTR = r"(?<=EEG )(.*)(?=-REF)"
            reLowC = ['FP1', 'FP2', 'FZ', 'CZ', 'PZ','T','A']
            if re.search(reSTR, i) and re.search(reSTR, i).group() in reLowC:
                lowC = i[0:5] + i[5].lower() + i[6:]
                mne.channels.rename_channels(EEGseries.info, {i: re.findall(reSTR, lowC)[0]})
            elif re.search(reSTR, i):
                mne.channels.rename_channels(EEGseries.info, {i: re.findall(reSTR, i)[0]})
            else:
                print(i)
        return EEGseries

    def annoTUH(self,annoPath=False, header=None):
        df = pd.read_csv(annoPath, sep=" ", skiprows=1, header=header)
        df.fillna('null', inplace=True)
        #df.dropna()
        #annoTUH = df[df[0].between(window[0], window[1]) | df[1].between(window[0], window[1])]
        return df

    def slidingWindow(self,edfInfo):
        """
        Split singal up into windows.
        """
        windowEEG = defaultdict(list)
        sampleWindow = self.tWindow*self.freq
        tN = edfInfo["rawData"].last_samp
        steps=int(sampleWindow*(1-self.Overlap))
        n=0
        for i in range(0, tN, steps):
            windowKey = f"window_{n}"

            windowEEG[windowKey] = self.DataMaker(edfInfo, t0=i, tWindow=sampleWindow)
            n+=1
        if (1+tN) % int(steps) != 0:
            pass #Trow away the last window comment out to get it back
            #windowKey = f"window_{n}"
            #windowEEG[windowKey] = self.DataMaker(edfInfo, t0=int(tN-sampleWindow), tWindow=sampleWindow)
        return windowEEG

    def DataMaker(self,EEGseries, t0, tWindow):
        """
        Extract data for a and lables for a window. If fourier transformations is neede this is the place to do it.
        """
        tStart=t0/self.freq
        tEnd=(t0+tWindow)/self.freq
        Data={}
        Data["stepval"] = [t0,t0+tWindow]
        Data["tval"] = [tStart,tEnd]
        Data["X"] = EEGseries["rawData"].get_data(start=int(t0), stop=int(t0+tWindow))
        df=EEGseries["Annotation"]
        #Data["Y"]= df.loc[df[0].between(tStart, tEnd) | df[1].between(tStart, tEnd)][2]
        iStart=sum(df[0]<=tStart)-1
        iEnd=sum(df[1]<=tEnd)
        if iStart==iEnd:
            #Assing lable as last started artifact.
            lable=df.iloc[iStart,2]
        else:
            #if and artifact end in the window assign lable to domenet artifact
            lable=df.iloc[[iStart,iEnd][np.argmax([tStart-df.iloc[iStart,0],tEnd-df.iloc[iStart,1]])],2]
        Data["lable"]=lable
        Data["Y"]=self.one_hot_eoncoding==lable
        if np.sum(Data["Y"]) !=1:
            raise Exception("To few or to many labels.")
        return Data

#loader=dataloader(Time_interval=1,Overlap=0)
#edfdict={"path": path}
#dataDict=loader.readRawEdf(edfdict)
#windowdict=loader.slidingWindow(dataDict)
#print(windowdict)
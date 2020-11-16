
import numpy as np

import os
from datetime import datetime
from mne.io import read_raw_edf
from collections import defaultdict
path=os.path.join(os.getcwd(),r"Data\s007_2012_07_25\00001154_s007_t000")
import pandas as pd



def readRawEdf(edfDict=None, saveDir='',
               read_raw_edf_param={'preload':True, "stim_channel":"auto"}):
    """
    Function to read the edf file in one given location.
    """
    edfDict["rawData"] = read_raw_edf(edfDict["path"]+".edf", **read_raw_edf_param)
    edfDict["Anotation"] = annoTUH(edfDict["path"]+r".tse")

    # comment this out to get meta data on recording time stamps, WARNING will given and error in python 3.7
    # tStart = edfDict["rawData"].annotations.orig_time-timedelta(hours=1)
    # tLast = int((1+edfDict["rawData"].last_samp)/edfDict["rawData"].info["sfreq"])
    # edfDict["t0"] = tStart
    # edfDict["tN"] = tStart + timedelta(seconds=tLast)

    edfDict["fS"] = edfDict["rawData"].info["sfreq"]
    return edfDict

def annoTUH(annoPath=False, window=[0,50000], header=None):
    df = pd.read_csv(annoPath, sep=" ", skiprows=1, header=header)
    df.fillna('null', inplace=True)
    #df.dropna()
    #annoTUH = df[df[0].between(window[0], window[1]) | df[1].between(window[0], window[1])]
    return df

def slidingWindow(edfInfo, tWindow, Overlap):
    """
    Split singal up into windows.
    tWindow is how long each window is in seconds
    Overlap given as a float eg: 0.25 for 0.25% overlap
    """
    windowEEG = defaultdict(list)
    sampleWindow = tWindow*edfInfo["fS"]
    tN = dataDict["rawData"].last_samp
    steps=int(sampleWindow*(1-Overlap))
    n=0
    for i in range(0, tN, steps):
        windowKey = f"window_{n}"

        windowEEG[windowKey] = DataMaker(edfInfo, t0=i, tWindow=sampleWindow)
        n+=1
    if (1+tN) % int(steps) != 0: #ToDO discus is this the right way to handle the last window
        windowKey = f"window_{n}"
        windowEEG[windowKey] = DataMaker(edfInfo, t0=int(tN-sampleWindow), tWindow=sampleWindow)
    return windowEEG

def DataMaker(EEGseries, t0, tWindow):
    """
    Extract data for a and lables for a window. If fourier transformations is neede this is the place to do it.
    """
    freq = EEGseries["fS"]
    tStart=t0/freq
    tEnd=(t0+tWindow)/freq
    Data={}
    Data["stepval"] = [t0,tWindow]
    Data["tval"] = [tStart,tWindow/freq]
    Data["X"] = EEGseries["rawData"].get_data(start=int(t0), stop=int(t0+tWindow))
    df=EEGseries["Anotation"]
    Data["Y"]= df.loc[df[0].between(tStart, tEnd) | df[1].between(tStart, tEnd)][2]
    return Data


edfdict={"path": path}
dataDict=readRawEdf(edfdict)
windowdict=slidingWindow(dataDict,tWindow=1,Overlap=0)
print(windowdict)
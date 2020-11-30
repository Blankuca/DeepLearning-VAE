import matplotlib.pyplot as plt
import numpy as np

class MNEPlotter:
    def __init__(self,CH_names,lableEncoding=[None]):
        if lableEncoding.all==[None]:
            self.decodeLable=False
        else:
            self.decodeLable=True
            self.decoding = lableEncoding


        self.CH_names=CH_names

    def plot(self,window,show=True):
        """
        Plots a window if given in out dataformat
        """
        data=window['X']
        y=np.arange(window['stepval'][0],window['stepval'][1])
        if self.decodeLable:
            lable=self.decoding[window['Y']==True]
        else:
            lable = window['Y']

        fig, axs=plt.subplots(len(self.CH_names),1,sharey=True,figsize=(6,20))
        for n,CH in enumerate(self.CH_names):
            axs[n].plot(y,data[n])
            axs[n].set_ylabel(CH)
        fig.suptitle(f"window betwen {window['tval'][0]} and {window['tval'][1]}S, lable {lable}")
        plt.show()




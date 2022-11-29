#import random
import numpy as np
#from dtw import *
from ont_fast5_api.fast5_interface import get_fast5_file
#from random import randrange
import itertools
from pathlib import Path
import h5py
import pandas as pd

import matplotlib.pyplot as plt


'''
returns list of numpy arrays representing reads as well as dataframe containing reads metadata.
'''
def getReads(f5_file="../basic_testdata/FAT10833_ab5a8994_951.fast5", withMetadata=False):
    reads = []
    if withMetadata:
        ids = []
        length = []

    if withMetadata:
        with get_fast5_file(f5_file, mode="r") as f5:
            for read in f5.get_reads():
                raw_data = read.get_raw_data()
                reads.append(raw_data)
                ids.append(read.read_id)
                length.append(len(raw_data))
                #min_rdata = min(raw_data)
                #max_rdata = max(raw_data)
                #print(read.read_id, raw_data, min_rdata, max_rdata)

        z = list(zip(ids, length))
        return reads, pd.DataFrame(z, columns=["id", "length"])

    else:
        with get_fast5_file(f5_file, mode="r") as f5:
            for read in f5.get_reads():
                raw_data = read.get_raw_data()
                reads.append(raw_data)

        return reads


def getRollingMin(sequence, window=50):
    rollingMin = []
    halfWindow = int(window/2)
    for i in range(halfWindow, len(sequence)-halfWindow):
        rollingMin.append(min(sequence[i-(halfWindow):i+(halfWindow)]))
    return rollingMin

def getRollingAvg(sequence, window=50):
    rollingAvg = []
    halfWindow = int(window/2)
    for i in range(halfWindow, len(sequence)-halfWindow):
        rollingAvg.append(sum(sequence[i-(halfWindow):i+(halfWindow)])/window)
    return rollingAvg
    


readsArr, metaData = getReads(withMetadata=True)

#print(readsArr)
#print(metaData)

sequence = readsArr[0][3000: 6000: 1]           #getting subsection of sequence


rollingWindow = 50
halfWindow = int(rollingWindow/2)
rollingMin = getRollingMin(sequence, rollingWindow)
rollingMinAvg = getRollingAvg(rollingMin, rollingWindow)            #average of the rolling min, for smoothing

rollingAvg = getRollingAvg(sequence, window=8)                      #average of the sequence vector, for smoothing

normalized = [a_i - b_i for a_i, b_i in zip(rollingAvg[46:len(rollingAvg)-46], rollingMinAvg)]          #sequence vector minus rolling min avg (expected base current fluctuation vector)

transform = np.fft.fft(sequence)

print(transform)

plt.plot([i for i in range(len(sequence))], sequence)                                           #sequence
#plt.plot([i for i in range(halfWindow, len(sequence)-halfWindow)], rollingMin)
#plt.plot([i for i in range(rollingWindow, len(sequence)-rollingWindow)], rollingMinAvg)
#plt.plot([i for i in range(4, len(sequence)-4)], rollingAvg)
plt.plot([i for i in range(rollingWindow, len(sequence)-rollingWindow)], normalized)

#plt.plot([i for i in range(3000)], transform)       #plotting real component of transform
#plt.plot([i for i in range(3000)], transform.imag)  #plotting imaginary component of transform
plt.show()


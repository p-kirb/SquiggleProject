import random
import math
import numpy as np
#from dtw import *
from ont_fast5_api.fast5_interface import get_fast5_file
#from random import randrange
import itertools
from pathlib import Path
import h5py
import pandas as pd

import matplotlib.pyplot as plt



def loadReads(f5_file="../basic_testdata/FAT10833_ab5a8994_951.fast5", withMetadata=False):
    '''
        returns list of numpy arrays representing reads as well as dataframe containing reads metadata.
    '''
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



def getClusters(sequence, noOfClusters, plot=False):
    '''
        calculates n centroids (where n=noOfClusters) around which data is clustered.
    '''

    prevCentroids = [0 for i in range(noOfClusters)]
    centroids = random.sample(sequence, noOfClusters)

    clusters = [[] for i in centroids]
    clustersAggregate = [0 for i in range(noOfClusters)]


    clusterIndex = 0
    distances = [0 for i in range(noOfClusters)]
    done = False

    while not done:
        clusters = [[] for i in centroids]
        clustersAggregate = [0 for i in range(noOfClusters)]
        for component in sequence:
            for i in range(noOfClusters):
                distances[i] = abs(centroids[i] - component)

            clusterIndex = distances.index(min(distances))
            clusters[clusterIndex].append(component)
            clustersAggregate[clusterIndex] += component

        prevCentroids = centroids.copy()
        for i in range(noOfClusters):
            centroids[i] = clustersAggregate[i]/len(clusters[i])


        if centroids == prevCentroids:
            done = True

    if plot:
        #plotting clusters
        for c in range(noOfClusters):
            plt.plot([i for i in range(len(sequence))], [centroids[c] for i in range(len(sequence))])



    return centroids

        

def getConvNormalisedSequence(sequence, rollingWindow=50):
    '''
        old normalisation method based on convolutions - has issues.
    '''
    rollingMin = getRollingMin(sequence, rollingWindow)
    rollingMinAvg = getRollingAvg(rollingMin, rollingWindow)            #average of the rolling min, for smoothing

    rollingAvg = getRollingAvg(sequence, window=8)                      #average of the sequence vector, for smoothing

    normalised = [a_i - b_i for a_i, b_i in zip(rollingAvg[46:len(rollingAvg)-46], rollingMinAvg)]          #sequence vector minus rolling min avg (expected base current fluctuation vector)

    return normalised


def getClustNormalisedSequence(sequence, noOfClusters):
    sequenceMin=min(sequence)
    centroids = getClusters(sequence, noOfClusters)
    newSequence = []
    differences = [0 for i in range(noOfClusters)]
    for component in sequence:
        for i in range(noOfClusters):
            differences[i] = abs(centroids[i]-component)

        newSequence.append(centroids[differences.index(min(differences))]-sequenceMin)
    
    return newSequence

        

if __name__ == "__main__":

    readsArr, metaData = loadReads(withMetadata=True)


    sequenceID = metaData["id"].iloc[76]
    print(f"sequence ID: {sequenceID}")
    sequence = readsArr[76][3000: 6000: 1].tolist()           #getting subsection of sequence
    plt.plot([i for i in range(len(sequence))], sequence)                                           #sequence


    #rollingWindow = 50
    #halfWindow = int(rollingWindow/2)

    #normalised = getConvNormalisedSequence(sequence, rollingWindow)     #has problems
    #plt.plot([i for i in range(rollingWindow, len(sequence)-rollingWindow)], normalised)

    noOfClusters = 4
    #centroids = getClusters(sequence, noOfClusters)
    #print(f"centroids:\n{centroids}")
    normalised = getClustNormalisedSequence(sequence, noOfClusters)
    plt.plot([i for i in range(len(normalised))], normalised)                                           #sequence

    #plt.plot([i for i in range(halfWindow, len(sequence)-halfWindow)], rollingMin)
    #plt.plot([i for i in range(rollingWindow, len(sequence)-rollingWindow)], rollingMinAvg)
    #plt.plot([i for i in range(4, len(sequence)-4)], rollingAvg)






    transform = np.fft.fft(sequence)
    #plt.plot([i for i in range(3000)], transform)       #plotting real component of transform
    #plt.plot([i for i in range(3000)], transform.imag)  #plotting imaginary component of transform
    plt.show()


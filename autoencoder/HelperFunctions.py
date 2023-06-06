import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
from Settings import datapath, subset


def loadReads():

    print("reading squiggles file...")
    reads = []
    ids = []
    with get_fast5_file(datapath, mode="r") as f5:
        for read in f5.get_reads():
            if read.read_id in subset:
                raw_data = read.get_raw_data()
                reads.append(raw_data)
                ids.append(read.read_id)

    return reads, ids


def zScoreNormalise(sequence):
    # print("z-score normalising")
    # start = time.time()
    
    zScoreNormalised = np.zeros(len(sequence))
    mean = np.mean(sequence)
    standardDev = np.std(sequence)
    zScoreNormalised = (sequence - mean) / standardDev
    # print(f"Took {time.time()-start} seconds.\n")

    return zScoreNormalised

def removeSpikes(sequence, spikeThresh=3):
    # print("removing spikes")
    # start = time.time()
    
    #spikeThresh = 2000
    returnList = np.array([i for i in sequence if abs(i) < spikeThresh])
    # print(f"Took {time.time()-start} seconds.\n")

    return returnList


def fullNormalise(sequence):
    '''
    1) spike removal
    2) z-score normalisation
    '''
    s = sequence.copy()
    s = zScoreNormalise(s)
    s = removeSpikes(s)

    return s
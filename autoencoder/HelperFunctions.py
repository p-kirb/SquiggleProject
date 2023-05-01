import numpy as np

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
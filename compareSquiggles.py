

#normalise all squiggles
#take random squiggle and compare it to all other squiggles to find the most similar



#for a given squiggle, see if there is overlap with any other squiggle
#   i.e. if any other squiggle contains something similar to the given squiggle
#   do a cross-correlation, passing the small section of the original vector over the test vector
#   resultant vector with the largest infinity norm is the most similar.


import pandas as pd
import numpy as np
import csv
import time
import h5py

def loadVectorsFromCSV():
    print("loading vectors from csv")
    start = time.time()
    with open("normalisedVectors.csv", "r") as f:
        vectors = list(csv.reader(f, delimiter=","))
    
    print(f"Took {time.time()-start} seconds.\n")
    return vectors

def loadVectorsFromHDF5():
    print("loading vectors from HDF5")
    start = time.time()
    f = h5py.File("normalisedVectors.hdf5", "r")

    vectors = f["vectors"]
    #print(vectors[0])
    
    print(f"Took {time.time()-start} seconds.\n")
    #TODO: decide how to deal with closing the file
    return vectors

def doCrossCorrelation(vec1, vec2):
    '''
        returns the infinity norm of the cross correlation (bigger = more similar)
    '''
    return np.amax(np.correlate(vec1, vec2, "full"))


def getSimilarities(vector, vectorsDS):
    '''
        do parallel comparison of all rows in dataframe
    '''
    print(f"comparing {len(vectorsDS)} vectors")
    start = time.time()

    similarities = np.empty(len(vectorsDS))
    for i in range(len(vectorsDS)):
        similarities[i] =doCrossCorrelation(vectorsDS[i], vector)
        if i % 100 == 0: print(i)

    print(f"Took {time.time()-start} seconds.\n")
    return similarities



if __name__ == "__main__":
    vectors = loadVectorsFromHDF5()

    vecToCompare = vectors[0][3000:6000]

    similarities = getSimilarities(vecToCompare, vectors)

    mostSimilarIndices = np.argsort(similarities)[::-1][:10]
    print(mostSimilarIndices)


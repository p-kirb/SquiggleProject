import json
from Settings import *
from HelperFunctions import loadReads
import math


with open("cSimilarties.json") as f:
    similarities = json.load(f)


filepath = "../../similar_testdata/similar_squiggles.fast5"

squiggles, ids = loadReads()

lengths = {ids[i]: len(squiggles[i]) for i in range(len(ids))}

'''
#summing distances as calculated both ways
bothWaysDict = {}
for i in range(len(ids)):
    bothWaysDict[ids[i]] = {}
    for j in range(len(ids)):
        if i != j:
            bothWaysDict[ids[i]][ids[j]] = similarities[ids[i]][ids[j]] + similarities[ids[j]][ids[i]]
similarities = bothWaysDict
'''
#scaling by log of length

for outerSquiggle, innerSquiggles in similarities.items():
    for id, length in lengths.items():
        if id in innerSquiggles:
            innerSquiggles[id] = innerSquiggles[id] * math.log(length)




for i, l in lengths.items():
    print(i, l)

print("\n\n")

#sorts and prints the squiggles ordered by euclidian distance to other squiggles (top squiggle in list is most similar)
for outerSquiggle, innerSquiggles in similarities.items():
    print(outerSquiggle)
    inner = sorted(innerSquiggles.items(), key=lambda x: x[1])
    for innerSquiggle, sim in inner:
        print("   ", innerSquiggle, sim)
import matplotlib.pyplot as plt


from normaliseSquiggle import loadReads


readsArr = loadReads()


squiggles = [3999, 1328, 1341]#, 1340, 1339, 1338, 1337, 1336, 1335, 1334]      #most similar squiggles to squiggle 1[3000:6000] - calculated using cross correlation similarity
fig, axs = plt.subplots(len(squiggles))
for i in range(len(squiggles)):
    #plt.figure()
    axs[i].plot([i for i in range(len(readsArr[squiggles[i]]))], readsArr[squiggles[i]])

plt.show()
import sphereicalCoordinates
import histograms
import numpy as np

#Returns an enormous Nx100 feature vector of bin votes and a Nx1 class vector of actions
#using the entire dataset excepting the test entry and the last (19th) entry, which has missing data
def setup(test):
    X = [0]*5426
    Y = np.zeros(5426)
    index = 0
    for i in range(0, 19):
        print(i)
        if (i != test):
            sperson = str(int(i/2)+1)
            strial = "0"+str(i%2+1)
            if (i < 18):
                sperson = "0"+sperson
            filename = 'joints\\joints_s' + sperson + '_e' + strial + '.txt'
            data = sphereicalCoordinates.prepData(sphereicalCoordinates.getData(filename), i)
            stats = histograms.statistics(data)
            #Every action
            for j in range(0, 10):
                #Every frame
                for k in range(0, len(data[j])):
                    #Compute a histogram representing a posture
                    histo = histograms.getHisto(data[j][k], stats[0][j], stats[1][j], stats[2][j], stats[3][j])
                    X[index] = histo
                    Y[index] = j
                    index+=1
    return [X, Y]

def meanVectors(X, Y):
    meanVec = []
    index = 0
    for i in range(0, 10):
        m = np.zeros(100)
        while(Y[index] < i):
            index+=1
        goOn = True
        start = index
        while((Y[index] == i) & (goOn)):
            m+=X[index]
            if (index < len(Y)):
                index+=1
            if (index == len(Y)):
                index-=1
                goOn = False
        if (goOn == False):
            index+=1
        meanVec.append((1/(index-start))*m)
    return meanVec
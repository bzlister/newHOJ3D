import sphereicalCoordinates
import histograms
import numpy as np

#Returns an enormous Nx100 feature vector of bin votes and a Nx1 class vector of actions
#using the entire dataset excepting the test entry and the last (19th) entry, which has missing data
#Thanks to Sebastian Raschka for his excellent LDA tutorial http://sebastianraschka.com/Articles/2014_python_lda.html
def setup(test):
    X = []
    Y = []
    index = 0
    for i in range(0, 19):
        if (i != test):
            sperson = str(int(i/2)+1)
            strial = "0"+str(i%2+1)
            if (i < 18):
                sperson = "0"+sperson
            filename = 'joints\\joints_s' + sperson + '_e' + strial + '.txt'
            print(filename)
            data = sphereicalCoordinates.prepData(sphereicalCoordinates.getData(filename), i)
            stats = histograms.statistics(data)
            #Every action
            for j in range(0, 10):
                #Every frame
                for k in range(0, len(data[j])):
                    #Compute a histogram representing a posture
                    histo = histograms.getHisto(data[j][k], stats[0][j], stats[1][j], stats[2][j], stats[3][j])
                    X.append(histo)
                    Y.append(j)
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


def withinClassScatter(X, Y, meanVec):
    index = 0
    S_W = np.zeros((100, 100))
    for cl, mv in zip(range(0, 10), meanVec):
        class_sc_mat = np.zeros((100, 100))
        while (Y[index] == cl):
            row = X[index].reshape(100, 1)
            mv = mv.reshape(100, 1)
            class_sc_mat += (row-mv).dot((row-mv).T)
            index+=1
        S_W += class_sc_mat
    return S_W

def betweenClassScatter(X, Y, meanVec):
    overall_mean = np.mean(X, axis=0).reshape(100, 1)
    S_B = np.zeros((100, 100))
    for i in range(0, 10):
        n = Y.count(i)
        m = meanVec[i].reshape(100, 1)
        S_B += n*(m - overall_mean).dot((m - overall_mean).T)
    return S_B
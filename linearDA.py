import sphereicalCoordinates
import histograms
import numpy as np
from matplotlib import pyplot as plt

#Returns an enormous Nx100 feature vector of bin votes and a Nx1 class vector of actions
#using the entire dataset excepting the test entry and the last (19th) entry, which has missing data
#Thanks to Sebastian Raschka for his excellent LDA tutorial http://sebastianraschka.com/Articles/2014_python_lda.html
def setup(test):
    X = np.array([0]*100)
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
                    X = np.vstack((X, histo))
                    Y.append(j)
                    index+=1
    X = np.delete(X, 0, 0)
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

def eigen(S_W, S_B, X):
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:,i].reshape(100,1)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    eigv_sum = sum(eig_vals)
    for i,j in enumerate(eig_pairs):
        print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
    
    W = np.hstack((eig_pairs[0][1].reshape(100,1), eig_pairs[1][1].reshape(100, 1), eig_pairs[2][1].reshape(100, 1), eig_pairs[3][1].reshape(100, 1), eig_pairs[4][1].reshape(100, 1), eig_pairs[5][1].reshape(100, 1)))
    X_lda = X.dot(W)
    return X_lda

def plotLDA(X_lda, Y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['black', 'red', 'maroon', 'yellow', 'olive', 'green', 'aqua', 'teal', 'fuchsia', 'navy']
    labels = ['walk', 'sitDown', 'standUp', 'pickUp', 'carry', 'throw', 'push', 'pull', 'waveHands', 'clapHands']
    markers = ['.', 'o', 'v', '^', '<', '>', '*', '+', 'x', '|']
    index = 0
    x = [0]*30
    y = [0]*30
    z = [0]*30
    for a in range(0, 30):
        x[a] = X_lda[a][0].real
        y[a] = X_lda[a][1].real
        z[a] = X_lda[a][2].real
    for a in range(0, 10):
        length = int(Y.count(a)/3)
        for c in range(0, 3):
            xSum = 0
            ySum = 0
            zSum = 0
            for b in range(index, index+length):
                xSum+=X_lda[b][0].real
                ySum+=X_lda[b][1].real
                zSum+=X_lda[b][2].real
            x[3*a+c] = xSum/length
            y[3*a+c] = ySum/length
            z[3*a+c] = zSum/length
            index+=length
    
    for i in range(0, 10):
        ax.scatter(x[3*i:3*i+3], y[3*i:3*i+3], z[3*i:3*i+3], marker=markers[i], color=colors[i], label=labels[i])

    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_zlabel('LD3')
    plt.show()

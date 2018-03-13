import math
import scipy.stats as st
import numpy as np

#Finds mean and standard deviation of alpha and theta for a specific joint, across all frames in a specific action
#[alpha/theta mean/deviations][action][joint]
def statistics(angles):
    alphaMean = [0]*10
    thetaMean = [0]*10
    for i in range(0, 10):
        alphaActionSum = [0]*9
        thetaActionSum = [0]*9
        for j in range(0, len(angles[i])):
            for k in range(0, 9):
                alphaActionSum[k]+=angles[i][j][k][0]
                thetaActionSum[k]+=angles[i][j][k][1]        
        alphaMean[i] = [(1/len(angles[i]))*x for x in alphaActionSum]
        thetaMean[i] = [(1/len(angles[i]))*x for x in thetaActionSum]
    alphaDev = [0]*10
    thetaDev = [0]*10
    for x in range(0, 10):
        alphaActionDev = [0]*9
        thetaActionDev = [0]*9
        for y in range(0, len(angles[x])):
            for z in range(0, 9):
                alphaActionDev[z]+=math.pow(angles[x][y][z][0]-alphaMean[x][z], 2)
                thetaActionDev[z]+=math.pow(angles[x][y][z][1]-thetaMean[x][z], 2)
        alphaDev[x] = [math.sqrt((1/len(angles[x]))*q) for q in alphaActionDev]
        thetaDev[x] = [math.sqrt((1/len(angles[x]))*q) for q in thetaActionDev]
    return [alphaMean, thetaMean, alphaDev, thetaDev]

#Returns a 100-bin histogram
#A histogram represents a single posture (frame in action)
def getHisto(joints, alphaMeans, thetaMeans, alphaDevs, thetaDevs):
    bins= [0]*10
    for b in range(0, 10):
        bins[b] = [0]*10
    delta = math.pi/10
    for j in range(0, 9):
        alphaStart = int(joints[j][0]/delta)
        thetaStart = int(joints[j][1]/delta)
        for n in range(max(alphaStart-1, 0), min(alphaStart+2, 10)):
            for m in range(max(thetaStart-1, 0), min(thetaStart+2, 10)):
                alphaVote = st.norm.cdf(((n+1)*delta - alphaMeans[j])/alphaDevs[j]) - st.norm.cdf((n*delta-alphaMeans[j])/alphaDevs[j])
                thetaVote = st.norm.cdf(((m+1)*delta - thetaMeans[j])/thetaDevs[j]) - st.norm.cdf((m*delta - thetaMeans[j])/thetaDevs[j])
                bins[n][m]+=alphaVote*thetaVote
    return np.array(bins).flatten().tolist()
import math
import scipy.stats as st

#Finds mean and standard deviation of alpha and theta for a specific joint, across all frames in a specific action
#[alpha/theta mean/deviations][action][joint]
def statistics(angles):
    alphaMean = [0]*10
    thetaMean = [0]*10
    for i in range(0, 10):
        actualFrames = 0
        alphaActionSum = [0]*9
        thetaActionSum = [0]*9
        for j in range(0, len(angles[i])):
            if (angles[i][j] != 0):
                actualFrames+=1
                for k in range(0, 9):
                    alphaActionSum[k]+=angles[i][j][k][0]
                    thetaActionSum[k]+=angles[i][j][k][1]
        
        alphaMean[i] = [(1/actualFrames)*x for x in alphaActionSum]
        thetaMean[i] = [(1/actualFrames)*x for x in thetaActionSum]
    alphaDev = [0]*10
    thetaDev = [0]*10
    for x in range(0, 10):
        actualFrames = 0
        alphaActionDev = [0]*9
        thetaActionDev = [0]*9
        for y in range(0, len(angles[x])):
            if (angles[x][y] != 0):
                actualFrames+=1
                for z in range(0, 9):
                    alphaActionDev[z]+=math.pow(angles[x][y][z][0]-alphaMean[x][z], 2)
                    thetaActionDev[z]+=math.pow(angles[x][y][z][1]-thetaMean[x][z], 2)
        alphaDev[x] = [math.sqrt((1/actualFrames)*q) for q in alphaActionDev]
        thetaDev[x] = [math.sqrt((1/actualFrames)*q) for q in thetaActionDev]
    return [alphaMean, thetaMean, alphaDev, thetaDev]

#Returns a 100-bin histogram with probability calculations in 9 or fewer bins, depending on the position of the joint
def getHisto(alpha, theta, alphaMean, thetaMean, alphaDev, thetaDev):
    bins= [0]*10
    for i in range(0, 10):
        bins[i] = [0]*10
    
    delta = math.pi/10
    alphaStart = int(alpha/delta)
    thetaStart = int(theta/delta)
    for n in range(max(alphaStart-1, 0), min(alphaStart+2, 10)):
        for m in range(max(thetaStart-1, 0), min(thetaStart+2, 10)):
            bins[n][m] = (st.norm.cdf(((n+1)*delta - alphaMean)/alphaDev) - st.norm.cdf((n*delta-alphaMean)/alphaDev))*(st.norm.cdf(((m+1)*delta - thetaMean)/thetaDev) - st.norm.cdf((m*delta - thetaMean)/thetaDev))
    return bins
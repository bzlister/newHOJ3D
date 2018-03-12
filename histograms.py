import math

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
                    print(x,y,z)
                    alphaActionDev[z]+=math.pow(angles[x][y][z][0]-alphaMean[x][z], 2)
                    thetaActionDev[z]+=math.pow(angles[x][y][z][1]-thetaMean[x][z], 2)
        
        alphaDev[x] = [math.sqrt((1/actualFrames)*q) for q in alphaActionDev]
        thetaDev[x] = [math.sqrt((1/actualFrames)*q) for q in thetaActionDev]
    return [alphaMean, thetaMean, alphaDev, thetaDev]
            
import math

#[alpha or theta][action][joint]
def statistics(angles):
    alphaSum = [0]*10
    thetaSum = [0]*10
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
        print(i, actualFrames)
        alphaSum[i] = [(1/actualFrames)*x for x in alphaActionSum]
        thetaSum[i] = [(1/actualFrames)*x for x in thetaActionSum]
    return [alphaSum, thetaSum]
            
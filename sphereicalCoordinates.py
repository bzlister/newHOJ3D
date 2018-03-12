import math
import os, os.path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import re
import numpy as np
import random as rand
import vectorMath as vm

#Returns [joint][coordinate][frame]
def getData(filename):
    table = pd.read_table(filename, header=None, delim_whitespace=True)
    data = {'frame': table[0],
            'hip center':[table[1], table[2], table[3]],
            'left hip': [table[25], table[26], table[27]],
            'right hip': [table[49], table[50], table[51]],
            'head': [table[10], table[11], table[12]],
            'left elbow': [table[16], table[17], table[18]],
            'right elbow': [table[28], table[29], table[30]],
            'left hand': [table[22], table[23], table[24]],
            'right hand': [table[34], table[35], table[36]],
            'left knee': [table[40], table[41], table[42]],
            'right knee': [table[52], table[53], table[54]],
            'left foot': [table[46], table[47], table[48]],
            'right foot': [table[58], table[59], table[60]]}
    return data

#Converts the data to sphereical coordinates and restructures with the following specification:
#[action][frame][joint][alpha, theta]
def prepData(data, person):
    joints = ['head', 'left elbow', 'right elbow', 'left hand', 'right hand', 'left knee', 'right knee', 'left foot', 'right foot']
    labels = getLabels(person)
    angleData = 10*[0]
    for j in range (0, 10):
        index = 0
        if (index < len(data['frame'])):
            lowFrame = labels[j][1]
            highFrame = labels[j][2]
            frameData = []          
            q = True
            while (q & (data['frame'][index] < lowFrame)):
                index+=1
                if (index == len(data['frame'])):
                    index-=1
                    q = False
            index2 = index
            while ((data['frame'][index] <= highFrame) & q):
                x0 = data['hip center'][0][index]
                y0 = data['hip center'][1][index]
                z0 = data['hip center'][2][index]
                xL = data['left hip'][0][index]
                xR = data['right hip'][0][index]
                zL = data['left hip'][2][index]
                zR = data['right hip'][2][index]
                mag = math.sqrt(math.pow(xL-xR, 2)+math.pow(zL-zR, 2))
                href = [(xL-xR)/mag, 0, (zL-zR)/mag]
                vref = [0, 1, 0]
                frameData.append([])
                for k in range (0, 9):
                    x = data[joints[k]][0][index] - x0
                    y = data[joints[k]][1][index] - y0
                    z = data[joints[k]][2][index] - z0
                    alpha = math.acos(vm.dotProduct([x,y,z], href)/vm.getMagnitude([x,0,z]))
                    theta = math.acos(vm.dotProduct([x,y,z], vref)/vm.getMagnitude([x,y,z]))
                    while ((alpha <= -math.pi) | (alpha >= math.pi)):
                        alpha+=-math.copysign(1, alpha)*2*math.pi
                    while ((theta <= -math.pi) & (theta >= math.pi)):
                        theta+=-math.copysign(1, theta)*2*math.pi
                    frameData[index-index2].append([alpha, theta])
                index+=1
                if (index == len(data['frame'])):
                    index-=1
                    q = False
            angleData[j] = frameData
    return angleData


#[action][frame1, frame2]
def getLabels(person):
    labels = 10*[0]
    lineNo = 0
    sperson = str(int(person/2)+1)
    strial = "0"+str(person%2+1)
    if (person < 18):
        sperson = "0"+sperson
    pattern = re.compile(r"s"+sperson+"_e"+strial+"$")
    with open ('actionLabel.txt', 'rt') as in_file:
        j=0
        for line in in_file:
            if ((lineNo >= 11*person) & (lineNo < 11*person+11)):
                if (pattern.search(line) == None):
                    if (line.split(" ")[1] != "NaN"):
                        labels[j] = [line.split(" ")[0], int(line.split(" ")[1]), int(line.split(" ")[2])]
                    j+=1
            lineNo+=1
    return labels

#Used by animate
def visualizeFrame(data):
    joints = ['head', 'left elbow', 'right elbow', 'left hand', 'right hand', 'left knee', 'right knee', 'left foot', 'right foot']
    x = [0]*9*len(data['frame'])
    y = [0]*9*len(data['frame'])
    z = [0]*9*len(data['frame'])
    for h in range(0, len(data['frame'])):
        for i in range(0, 9):
            x[h*9+i] = data[joints[i]][0][h] - data['hip center'][0][h]
            y[h*9+i] = data[joints[i]][1][h] - data['hip center'][1][h]
            z[h*9+i] = data[joints[i]][2][h] - data['hip center'][2][h]
    return [x,y,z,data['frame']]

#Animates the skeleton sequence; used for visualization purposes
def animate(filename):


    first = getData(filename)
    a = visualizeFrame(first)
    t = np.array([np.ones(9)*i for i in range(len(first['frame']))]).flatten()
    df = pd.DataFrame({"time": t ,"x" : a[0], "y" : a[1], "z" : a[2]})
    fig = plt.figure()
    hax = fig.add_subplot(111, projection='3d')
    data=df[df['time']==0]
    colors = ['k', 'c', 'r', 'g', 'b', 'y']
    label = [1, 2, 2, 3, 3, 4, 4, 5, 5]
    thetaX = np.zeros(10)
    thetaY = np.linspace(0, 2, 10)
    thetaZ = np.zeros(10)
    hax.plot(thetaX, thetaY, thetaZ)

    hipLX = first['left hip'][0]
    hipLZ = first['left hip'][2]
    hipRX = first['right hip'][0]
    hipRZ = first['right hip'][2]
    mag = math.sqrt(math.pow(hipLX[0]-hipRX[0], 2) + math.pow(hipLZ[0]-hipRZ[0], 2))

    alphaX = ((hipLX[0]-hipRX[0])/mag)*np.linspace(0, 2, 100)
    alphaY = np.zeros(100)
    alphaZ = ((hipLZ[0]-hipRZ[0])/mag)*np.linspace(0, 2, 100)
    gruph = hax.scatter(alphaX, alphaY, alphaZ, c='r', marker='.')

    graph = hax.scatter(data.x, data.y, data.z, c=label, cmap=matplotlib.colors.ListedColormap(colors))

    hax.auto_scale_xyz([-1, 3], [-1, 3], [-1, 3])
    
    def update_graph(num):
        data=df[df['time']==num]
        graph._offsets3d = (data.x, data.y, data.z)
        
        num2 = int(num/9)
        mag = math.sqrt(math.pow(hipLX[num2]-hipRX[num2], 2) + math.pow(hipLZ[num2]-hipRZ[num2], 2))

        alphaX = ((hipLX[num2]-hipRX[num2])/mag)*np.linspace(0, 2, 100)
        alphaZ = ((hipLZ[num2]-hipRZ[num2])/mag)*np.linspace(0, 2, 100)
        gruph._offsets3d = (alphaX, alphaY, alphaZ)

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, len(first['frame'])*9-1, 
                                interval=50, blit=False)
    hax.set_xlabel('x label')
    hax.set_ylabel('y label')
    hax.set_zlabel('z label')
    plt.show()
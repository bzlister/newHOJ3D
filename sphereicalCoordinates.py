import math
import os, os.path
import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation

#Returns the data of the 12 tracked joints. For instance, data['hip center'][0] returns
#a list of x-coordinates of the hip center joint in every frame
def getData(filename):
    table = pandas.read_table(filename, header=None, delim_whitespace=True)
    data = {'hip center':[table[1], table[2], table[3]],
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

#Used with animation
def visualizeFrame(data):
    joints = ['hip center', 'left hip', 'right hip', 'head', 'left elbow', 'right elbow', 'left hand', 'right hand', 'left knee', 'right knee', 'left foot', 'right foot']

    x = [0]*12*560
    y = [0]*12*560
    z = [0]*12*560
    for h in range(0, 560):
        for i in range(0, 12):
            x[h*12+i] = data[joints[i]][0][h]
            y[h*12+i] = data[joints[i]][1][h]
            z[h*12+i] = data[joints[i]][2][h]
    return [x,y,z]

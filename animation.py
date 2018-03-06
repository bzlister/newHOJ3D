import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd
import sphereicalCoordinates


#Animates the joint skeleton

first = sphereicalCoordinates.getData("joints\\joints_s01_e01.txt")
a = sphereicalCoordinates.visualizeFrame(first)
t = np.array([np.ones(12)*i for i in range(560)]).flatten()
df = pd.DataFrame({"time": t ,"x" : a[0], "y" : a[2], "z" : a[1]})
def update_graph(num):
    data=df[df['time']==num]
    graph._offsets3d = (data.x, data.y, data.z)
    print(len(data.x))
    title.set_text('3D Test, time={}'.format(num))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

data=df[df['time']==0]
colors = ['k', 'c', 'r', 'g', 'b', 'y']
label = [0, 0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5]

graph = ax.scatter(data.x, data.y, data.z, c=label, cmap=matplotlib.colors.ListedColormap(colors))
ax.set_xlabel('X Label')
ax.set_ylabel('Z Label')
ax.set_zlabel('Y Label')
ax.auto_scale_xyz([-1, 3], [-1, 3], [-1, 3])
ani = matplotlib.animation.FuncAnimation(fig, update_graph, 560*12-1, 
                               interval=100, blit=False)

plt.show()
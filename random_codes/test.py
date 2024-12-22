#import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#
## Create a grid of points
#x = np.linspace(-5, 5, 100)
#y = np.linspace(-5, 5, 100)
#x, y = np.meshgrid(x, y)
#z = np.sin(np.sqrt(x**2 + y**2))  # Example function to create a surface
#
## Create a 3D plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
## Plot the points as a scatter plot
#ax.scatter(x, y, z, color='blue')
#
## Add labels and title
#ax.set_xlabel('X-axis')
#ax.set_ylabel('Y-axis')
#ax.set_zlabel('Z-axis')
#ax.set_title('3D Scatter Plot')
#
## Show plot
#plt.show()


import pandas as pd
import numpy as np

x = []
y = []
for i in range(20):
    x.append(i)


for i in range(20):
    if i % 2 == 0:
        y.append(2*i)
    else:
        y.append(-2*i)


correlation_matrix = np.corrcoef(x, y)
#correlation = correlation_matrix[0, 1]  # Extract correlation coefficient from the matrix
print("Correlation coefficient:", correlation_matrix[0, 1])
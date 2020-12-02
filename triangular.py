'''
======================
Triangular 3D surfaces
======================

Plot a 3D surface with a triangular mesh.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
scores = pd.read_csv('scores_fitness_based500_50.csv', header=None)
scores = np.unique(scores, axis=0)
print(scores)

X = scores[:,0]
Y = scores[:,1]
Z = scores[:,2]

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(X, Y, Z, 'o')
ax.plot_trisurf(X, Y, Z, cmap=plt.cm.Spectral)
ax.set_xlabel('cost (Baht)')
ax.set_ylabel('time (days)')
ax.set_zlabel('Mx^2 (man^2)')

plt.show()




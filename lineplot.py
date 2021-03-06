import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

plt.rcParams.update({'font.size': 22})
scores = pd.read_csv('scores.csv', header=None)
scores = np.unique(scores, axis=0)
print(scores)

X = scores[:,0]
Y = scores[:,1]
Z = scores[:,2]

tck, u = interpolate.splprep([X,Y,Z], s=2)
# x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
u_fine = np.linspace(0,1,200)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

fig2 = plt.figure(2)
fig2.tight_layout()
ax = fig2.add_subplot(111, projection='3d')
# ax.plot(x_true, y_true, z_true, 'b')
ax.plot(X, Y, Z, 'o')
# ax.plot(x_knots, y_knots, z_knots, 'go')
ax.plot(x_fine, y_fine, z_fine, 'b')
ax.set_title('Pareto front')
ax.set_xlabel('cost (Baht)', labelpad=22)
ax.set_ylabel('time (days)', labelpad=22)
ax.set_zlabel('Mx (man $^{2}$)', labelpad=40)

ax.tick_params(axis='z', which='major', pad=22)

fig2.show()
plt.show()
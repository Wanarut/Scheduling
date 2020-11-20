import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

result = pd.read_csv('result.csv')

X = result['X']
Y = result['Y']
Z = result['Z']

tck, u = interpolate.splprep([X,Y,Z], s=2)
# x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
u_fine = np.linspace(0,1,200)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

fig2 = plt.figure(2)
ax = fig2.add_subplot(111, projection='3d')
# ax.plot(x_true, y_true, z_true, 'b')
ax.plot(X, Y, Z, 'o')
# ax.plot(x_knots, y_knots, z_knots, 'go')
ax.plot(x_fine, y_fine, z_fine, 'b')
ax.set_xlabel('cost (Baht)')
ax.set_ylabel('time (days)')
ax.set_zlabel('Mx^2 (man^2)')
fig2.show()
plt.show()
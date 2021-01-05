import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

plt.rcParams.update({'font.size': 22})

scores = pd.read_csv('scores.csv', header=None)
scores = np.unique(scores, axis=0)
print(scores)

x = scores[:,0]
y = scores[:,1]
z = scores[:,2]

tck, u = interpolate.splprep([x,y,z], s=2)
u_fine = np.linspace(0,1,200)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

fig, axs = plt.subplots(3, 1)
fig.tight_layout()
axs[0].plot(x, y, 'o')
axs[0].plot(x_fine, y_fine, 'b')
axs[0].set_title('cost-time Fitnesses')
axs[0].set(xlabel='cost (Baht)', ylabel='time (days)')
# axs[1, 1].plot(y, z, 'tab:orange')
axs[1].plot(y, z, 'o')
axs[1].plot(y_fine, z_fine, 'r')
axs[1].set_title('time-Mx Fitnesses')
axs[1].set(xlabel='time (days)', ylabel='Mx (man $^{2}$)')
# axs[1, 0].plot(x, z, 'tab:green')
axs[2].plot(x, z, 'o')
axs[2].plot(x_fine, z_fine, 'g')
axs[2].set_title('cost-Mx Fitnesses')
axs[2].set(xlabel='cost (Baht)', ylabel='Mx (man $^{2}$)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

# fig2 = plt.figure(2)
# ax = fig2.add_subplot(111)
# ax.plot(x_true, y_true, z_true, 'b')
# ax.plot(X, Y, 'o')
# ax.plot(x_knots, y_knots, z_knots, 'go')
# ax.plot(x_fine, y_fine, 'b')
# ax.set_xlabel('cost (Baht)')
# ax.set_ylabel('time (days)')
# ax.set_zlabel('Mx^2 (man^2)')
# fig2.show()
plt.show()
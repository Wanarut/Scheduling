# import pandas as pd

# s_date = 'September 1, 2020 8:00 AM'
# f_date = 'October 1, 2020 5:00 AM'


# def days_diff(d1, d2, format='%B %d, %Y %I:%M %p'):
#     d1 = pd.to_datetime(d1, format=format)
#     d2 = pd.to_datetime(d2, format=format)
#     return (d2-d1).ceil('d').days


# # print(days_diff(s_date, f_date))

# Project_Start = pd.to_datetime(
#     'October 17, 2018 5:00 PM', format='%B %d, %Y %I:%M %p')
# print(Project_Start)
# Project_Start = Project_Start + pd.to_timedelta(10, unit='d')
# print(Project_Start)
# Project_Start = Project_Start - pd.to_timedelta('+10 days')
# print(Project_Start)

# # max_date = max(pd.to_datetime(tasks['Finish_Date'], format='%B %d, %Y %I:%M %p'))

# # predecessor = '8FS+150 days'
# # FS_loc = str(predecessor).find('FS')
# # print(FS_loc)

# # h_set = str('35,40').split(',')
# # h_set = list(map(int, h_set))
# # print(h_set)

# print(round(1.7))
# exit()

import numpy as np
# s = np.random.randint(10, size=100)
# print(s)

# scores = np.zeros(4, int)
# print(10**2)

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
result = pd.read_csv('result.csv')

X = result['Y']
Y = result['Z']
Z = result['X']

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(X, Y, Z, 'o')
ax.plot_trisurf(X, Y, Z, cmap=plt.cm.Spectral)
ax.set_xlabel('time (days)')
ax.set_ylabel('Mx^2 (man^2)')
ax.set_zlabel('cost (Baht)')

plt.show()




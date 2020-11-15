import pandas as pd

s_date = 'September 1, 2020 8:00 AM'
f_date = 'October 1, 2020 5:00 AM'


def days_diff(d1, d2, format='%B %d, %Y %I:%M %p'):
    d1 = pd.to_datetime(d1, format=format)
    d2 = pd.to_datetime(d2, format=format)
    return (d2-d1).ceil('d').days


# print(days_diff(s_date, f_date))

Project_Start = pd.to_datetime(
    'October 17, 2018 5:00 PM', format='%B %d, %Y %I:%M %p')
print(Project_Start)
Project_Start = Project_Start + pd.to_timedelta(10, unit='d')
print(Project_Start)
Project_Start = Project_Start - pd.to_timedelta('+10 days')
print(Project_Start)

# max_date = max(pd.to_datetime(tasks['Finish_Date'], format='%B %d, %Y %I:%M %p'))

# predecessor = '8FS+150 days'
# FS_loc = str(predecessor).find('FS')
# print(FS_loc)

# h_set = str('35,40').split(',')
# h_set = list(map(int, h_set))
# print(h_set)

import numpy as np
s = np.random.randint(10, size=100)
print(s)

scores = np.zeros(4, int)
print(10**2)
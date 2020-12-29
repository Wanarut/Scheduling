import pandas as pd
from sklearn import preprocessing
import numpy as np

xlsx = pd.ExcelFile('gen500_normal/scores_log.xlsx')
csv = pd.read_csv('gen500_normal/scores.csv', header=None)

distances = []
for sheet in xlsx.sheet_names:
    print(sheet)
    scores = pd.read_excel(io=xlsx, sheet_name=sheet, header=None)
    scores = scores[:70]
    # print(scores)
    
    # x = scores.values #returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # scores = pd.DataFrame(x_scaled)
    # print(scores)
    # exit()

    # distance = scores[0].pow(2) + scores[1].pow(2) + scores[2].pow(2)
    distance = scores[2]
    dis_mean = distance.mean()
    # print(dis_mean)
    distances.append(dis_mean)

distances.append(csv[2].mean())
np_distance = np.array(distances)
print(np_distance)
np.savetxt('distances.csv', np_distance.T, delimiter=',')
    
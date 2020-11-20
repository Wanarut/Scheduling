import numpy as np
scores = np.array(((79154978,724,1011454),(80212125,735,996754),(80019916,733,996918),(79923812,732,999812),(79635499,729,1002056)))
print(scores)
scores = scores[np.argsort(scores[:, 1])]
print(scores)
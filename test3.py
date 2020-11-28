import numpy as np
scores = np.array(((79154978,724,1011454),(80212125,735,996754),(80019916,733,996918),(79923812,732,999812),(79635499,729,1002056)))
# print(scores)
scores = scores[np.argsort(scores[:, 1])]
# print(abs(scores+scores)/2)
fitness_1 = 0
fitness_2 = 15
parent_1 = np.array(((0,1),(1,1),(2,1),(3,1),(4,1),(5,1)))
# parent_2 = np.array(((0,0),(1,1),(2,1),(3,1),(4,1),(5,1)))
parent_2 = np.array(((5,0),(4,1),(3,1),(2,1),(1,1),(0,1)))

print('parent_1', parent_1)
print('parent_2', parent_2)
center = (parent_1 + parent_2)/2
print('center', center)
diff = abs(parent_2 - parent_1)/2
print('diff', diff)
child_1 = center - ((fitness_1+1)/(fitness_1+fitness_2+1))*diff
child_2 = center + ((fitness_2+1)/(fitness_1+fitness_2+1))*diff
child_1 = np.rint(child_1)
child_2 = np.rint(child_2)

print('child_1', child_1)
print('child_2', child_2)
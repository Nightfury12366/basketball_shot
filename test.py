import numpy as np

m3 = []
m2_1 = []
m2_2 = []
a = [1, 2, 3]
b = [4, 5, 6]
c = [0, 0, 0]
d = [0, 0, 0]
m2_1.append(a)
m2_1.append(b)
m2_1 = np.array(m2_1)
m2_2.append(c)
m2_2.append(d)
m2_2 = np.array(m2_2)
m3.append(m2_1)
m3.append(m2_2)
m3 = np.array(m3)
print(m2_1[:, 0])

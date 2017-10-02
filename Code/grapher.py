import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


c = [0.01, 0.05, 0.5, 0.1, 10, 50, 100, 0.01, 0.05, 0.5, 0.1, 10, 50, 100, 0.01, 0.05, 0.5, 0.1, 10, 50, 100, 0.01, 0.05, 0.5, 0.1, 10, 50, 100]
y = [11.18530885, 11.18530885, 14.57985531, 11.18530885, 58.59766277, 58.59766277, 58.59766277, 97.38452977, 97.38452977, 97.38452977, 97.38452977, 97.38452977, 97.38452977, 97.38452977, 10.1836394, 10.1836394, 10.1836394, 10.1836394, 10.1836394, 10.1836394, 10.1836394, 95.9933222, 96.10461881, 95.9933222, 96.04897051, 95.9933222, 95.9933222, 95.9933222]

j = plt.figure(1)
j.suptitle('Support Vector Machine \n (Different C values)')

plt.title('Optical Character Recogniser')
plt.plot(c, y, 'ro')
plt.xlabel('C')
plt.ylabel('Predicted Percent')
plt.show()


y1 = [58.19672131, 58.19672131, 58.19672131, 58.19672131]
y2 = [58.19672131, 58.19672131, 59.83606557, 59.01639344]
y3 = [58.19672131, 58.19672131, 58.19672131, 58.19672131]
y4 = [60.6557377, 77.04918033, 93.44262295, 87.70491803]
z1 = [0.01, 0.05, 0.5, 0.1]
z2 = [0.01, 0.05, 0.5, 0.1]
z3 = [0.01, 0.05, 0.5, 0.1]
z4 = [0.01, 0.05, 0.5, 0.1]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(y1, z1, label='RBF')
ax.plot(y2, z2, label='Poly')
ax.plot(y3, z3, label='Sigmoid')
ax.plot(y4, z4, label='Linear')
plt.title('Product Review Analysis')
ax.set_xlabel('Kernel', )
ax.set_ylabel('Predicted Percent')
plt.legend(loc=2)
plt.show()


x1 = [85.7540345, 85.14190317, 85.08625487, 85.19755147, 85.19755147, 85.19755147]
x2 = [84.75236505, 85.6983862, 86.47746244, 85.92097941, 84.86366166, 88.36950473]
x3 = [91.98664441, 94.15692821, 96.10461881, 96.82804674, 95.43683918, 85.7540345]
plt.figure()
plt.title('Boosting OCR')
plt.plot(x1, label='Max Depth: 10')
plt.plot(x2, label='Max Depth: 20')
plt.plot(x3, label='Max Depth: 30')
plt.legend(loc=2)
plt.show()

x1 = [0.01, 0.05, 0.1, 0.5, 1, 1.5]
x2 = [0.01, 0.05, 0.1, 0.5, 1, 1.5]
x3 = [0.01, 0.05, 0.1, 0.5, 1, 1.5]
y1 = [84.42622951, 90.16393443, 90.98360656, 91.80327869, 91.80327869, 91.80327869]
y2 = [91.80327869, 92.62295082, 92.62295082, 90.16393443, 90.98360656, 90.98360656]
y3 = [92.62295082, 90.98360656, 92.62295082, 91.80327869, 93.44262295, 93.44262295]
plt.figure()
plt.title('Boosting Product Review')
plt.plot(x1, y1, label='Max Depth: 10')
plt.plot(x2, y2, label='Max Depth: 20')
plt.plot(x3, y3, label='Max Depth: 30')
plt.legend(loc=2)
plt.show()

x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y = [92.62295082, 92.62295082, 92.62295082, 92.62295082, 92.62295082, 92.62295082, 92.62295082, 92.62295082, 92.62295082, 92.62295082]
plt.title('KNN Product Review')
plt.plot(x, y)
plt.show()

x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y = [97.82971619, 97.10628826, 96.82804674, 96.60545353, 96.38286032, 95.8263773, 95.38119087, 94.88035615, 94.54646633, 94.54646633]
plt.title('KNN OCR')
plt.plot(x, y)
plt.show()

x = [1, 2, 3, 4, 5, 6]
y = [79, 64.86, 75.57, 65.95, 77.62, 76.62]
l = ['Decision Tree Classifier', 'Neural Networks', 'Support Vector Machines', 'Random Forest Classification',\
     'k Nearest Neighbours', 'Gradient Boosting Classifier']
plt.xticks(x, l)
plt.title("Algorithms Vs Accuracy")
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.plot(x, y)
for i, j in zip(x, y):
     plt.annotate(str(j), xy=(i,j))
plt.show()
#Guide
#In this project, we will use perceptrons to model the fundamental building blocks of computers — logic gates.
#diagrams of AND, OR, and XOR gates
#Input 1	Input 2	Output
#0	0	0
#0	1	0
#1	0	0
#1	1	1

#We’ll discuss how an AND gate can be thought of as linearly separable data and train a perceptron to perform AND.

#Input 1	Input 2	Output
#0	0	0
#0	1	1
#1	0	1
#1	1	0

#We’ll think about why an XOR gate isn’t linearly separable and show how a perceptron fails to learn XOR



import seaborn as sns
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
data = [[0, 0],[0,1],[1,0], [1,1]]
labels = [0, 0, 0,1]
plt.scatter([point[0] for point in data],[point[1] for point in data],c = labels)
plt.show()
classifier = Perceptron(max_iter = 40)
classifier.fit(data,labels)
print(classifier.score(data,labels))
#print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
point_grid = list(product(x_values,y_values))
distances = classifier.decision_function(point_grid)
abs_distances = [abs(pt) for pt in distances]
distances_matrix = np.reshape(abs_distances, (100,100))
heatmap = plt.pcolormesh(x_values,y_values,distances_matrix)
plt.colorbar(heatmap)
plt.show()















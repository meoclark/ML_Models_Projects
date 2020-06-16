import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
#print(digits.DESCR)
#print(digits.data)
#print(digits.target)

plt.gray() 

plt.matshow(digits.images[100])

plt.show()
print(digits.target[100])
# Figure size (width, height)

fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images

for i in range(64):

    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position

    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])

    # Display an image at the i-th position

    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value

    ax.text(0, 7, str(digits.target[i]))

plt.show()
from sklearn.cluster import KMeans
model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

new_samples = np.array( [
[0.00,0.99,2.06,2.29,1.07,0.00,0.00,0.00,0.00,7.16,7.62,7.62,6.79,1.07,0.00,0.00,0.00,0.99,1.37,1.75,7.17,6.86,1.45,0.00,0.00,0.00,0.00,0.00,1.22,6.86,6.40,0.00,0.00,0.00,0.00,0.00,0.00,3.28,7.62,1.68,0.00,0.00,0.00,0.00,0.00,3.13,7.62,3.05,0.46,3.51,4.96,5.34,5.72,7.63,7.09,1.60,2.97,7.62,7.62,7.47,6.49,7.55,7.62,7.40],
[0.00,1.07,3.36,3.81,2.75,0.15,0.00,0.00,0.15,6.56,7.63,6.86,7.55,4.88,0.15,0.00,2.44,7.62,2.82,0.00,4.80,7.62,6.25,0.23,4.57,7.17,0.23,0.00,0.08,4.12,7.62,0.77,4.57,6.86,0.00,0.00,0.00,3.05,7.62,0.76,4.57,6.86,0.00,0.00,0.00,3.05,7.62,0.76,3.97,7.62,6.40,6.10,6.10,6.71,7.62,0.69,0.15,3.28,4.57,4.57,4.57,4.58,2.75,0.00],
[0.15,5.11,7.62,7.62,7.62,6.94,2.36,0.00,0.23,5.64,3.66,2.29,2.59,6.26,7.55,1.22,0.00,0.00,0.00,0.00,0.00,1.60,7.62,2.29,0.00,0.00,0.00,0.00,0.00,2.37,7.62,1.75,0.00,2.29,3.74,2.44,2.13,6.41,7.09,0.30,0.00,5.18,7.62,7.62,7.62,7.63,2.52,0.00,0.00,3.81,7.62,6.94,7.63,7.55,6.86,4.73,0.00,1.52,4.58,4.57,4.57,4.57,4.04,2.29],
[0.00,0.00,0.00,0.99,1.52,1.60,2.21,0.15,0.00,1.45,6.02,7.62,7.62,7.62,7.62,5.34,0.92,7.17,6.79,3.20,1.60,1.15,4.65,7.63,5.19,7.24,0.99,0.00,0.00,0.00,4.50,7.55,6.86,4.88,0.00,0.00,0.00,0.30,7.17,5.26,6.86,6.25,2.90,0.61,0.00,4.96,7.55,1.68,3.58,6.95,7.62,7.62,7.62,7.62,3.97,0.00,0.00,0.00,1.45,3.05,3.05,2.75,0.00,0.00]
]  )

new_labels = model.predict(new_samples)

print(new_labels) 
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
    
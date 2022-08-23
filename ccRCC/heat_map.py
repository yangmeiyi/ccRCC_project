import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# x1 = [[82.9, 83.2, 83.1, 82.5, 83.3],
#      [83.3, 84.7,84.7,84.8,87.0],
#      [81.1,83.8,80.7,81.9,82.5],
#      [82.4,84.6,81.5,81.7,84.6],
#      [83.1,86.5,82.6,85.4,82.8]]
#
# x2 = [[86.4,91.1,86.5,85.8,84.7],
#       [90.2,89.4,89.5,89.5,92.0],
#       [86.4,88.4,83.6,90.2,85.4],
#       [85.7,89.6,85.1,86.2,88.5],
#       [84.6,91.5,85.3,84.3,86.2]]

x1 = [[72.9,78.2,73.1,72.5,73.3],
     [78.3,76.7,76.7,78.8,80.0],
     [73.1,78.8,69.4,71.9,72.5],
     [72.4,78.6,71.5,71.7,75.6],
     [73.1,80.1,72.6,75.4,72.8]]

x2 = [[76.4,83.1,76.5,75.8,74.7],
      [83.2,81.4,81.5,81.5,85.0],
      [76.4,81.4,69.8,70.2,72.4],
      [75.7,81.6,70.1,72.2,74.5],
      [74.6,84.9,72.3,74.3,73.2]]

x1 = np.array(x1)
x2 = np.array(x2)

f, (ax1, ax2) = plt.subplots(figsize=(6, 6), nrows=2)

sns.heatmap(x1, annot=True, fmt='.1f', ax=ax1)
sns.heatmap(x2, annot=True, fmt='.1f', ax=ax2)

ax1.set_title('Accuracy (ACC)')
ax2.set_title('Area Under Curve (AUC)')
ax1.set_xticklabels(['Roation','Affine', 'Crop', 'Blur','Noise' ])
ax1.set_yticklabels(['Roation','Affine', 'Crop', 'Blur','Noise'], rotation=45)
ax2.set_xticklabels(['Roation','Affine', 'Crop', 'Blur','Noise' ])
ax2.set_yticklabels(['Roation','Affine', 'Crop', 'Blur','Noise' ], rotation=45)
plt.tight_layout(h_pad=1.6)
plt.savefig("./heat_map_cnn.pdf")

plt.show()

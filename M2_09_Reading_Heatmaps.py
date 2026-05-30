import seaborn as sns
import matplotlib.pyplot as plt

### Sample data for the heatmap
data = [
    [0.8, 0.2, 0.4, 0.6],
    [0.3, 0.5, 0.7, 0.1],
    [0.9, 0.3, 0.6, 0.2],
    [0.2, 0.4, 0.8, 0.5]
]

### Create the heatmap using Seaborn
sns.heatmap(data,
            annot=False,
            cmap='coolwarm',
            linewidths=0.5,
            linecolor='gray',
            cbar=True,
            cbar_kws={"orientation": "vertical"},
            square=True,
            fmt='.2f',
            xticklabels=['A', 'B', 'C', 'D'],
            yticklabels=['W', 'X', 'Y', 'Z'])

### Set title and labels
plt.title('Sample Heatmap')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

### Display the heatmap
plt.show()

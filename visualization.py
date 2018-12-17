import utils
import seaborn as sns
import matplotlib.pyplot as plt


_, _, data = utils.get_data2()
print(data)

# sns.pairplot(data, hue="Size", size=3, kind='scatter').add_legend()
# sns.heatmap(data.corr())

# plt.imshow(data.corr(), cmap='hot')
# plt.show()

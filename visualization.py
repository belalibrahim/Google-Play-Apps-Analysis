import utils
import seaborn as sns
import matplotlib.pyplot as plt


predictors, response = utils.get_data_2()
print(predictors)
# sns.pairplot(data, hue="Size", size=3, kind='scatter').add_legend()
#sns.heatmap(data.corr())

# plt.imshow(data.corr(), cmap='hot')
#plt.show()

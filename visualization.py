import utils
import seaborn as sns
import matplotlib.pyplot as plt


_, _, data = utils.get_data2()

# Rug plot
#sns.stripplot(x="Category_c", y="Rating", data=data, jitter=True)
#sns.stripplot(x="Price", y="Rating", data=data, jitter=True)

# sns.heatmap(data.corr())
# Scatter Plot
'''
sns.jointplot(x="Category_c", y="Rating", data=data)
sns.jointplot(x="Price", y="Rating", data=data)
sns.jointplot(x="Genres_c", y="Rating", data=data)
sns.jointplot(x="Price", y="Rating", data=data)
sns.jointplot(x="Type", y="Rating", data=data)
sns.jointplot(x="Installs", y="Rating", data=data)
sns.jointplot(x="Size", y="Rating", data=data)
'''
'''sns.distplot(data['Rating'])
sns.distplot(data['Category_c'])
#sns.distplot(data['Genres_c'])
#sns.distplot(data['Price'])
#sns.distplot(data['Size'])'''
'''sns.boxplot(x="Category_c", y="Rating", data=data)
sns.boxplot(x="Price", y="Rating", data=data)
sns.boxplot(x="Genres_c", y="Rating", data=data)
sns.boxplot(x="Price", y="Rating", data=data)
sns.boxplot(x="Category_c", y="Rating", data=data)
sns.boxplot(x="Installs", y="Rating", data=data)
#sns.boxplot(x="Size", y="Rating", data=data)'''
# plt.imshow(data.corr(), cmap='hot')
plt.show()

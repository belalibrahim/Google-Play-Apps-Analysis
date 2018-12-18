import utils
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.

import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
_, _, data = utils.get_data2()
'''
# rating distibution
rcParams['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(data.Rating, color="Red", shade = True)
g.set_xlabel("Rating")
g.set_ylabel("Frequency")
plt.title('Distribution of Rating',size = 20)

g = sns.countplot(x="Category",data=data, palette = "Set1")
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

plt.title('Count of app in each category',size = 20)
'''
# rating distibution
rcParams['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(data.Reviews, color="Green", shade = True)
g.set_xlabel("Reviews")
g.set_ylabel("Frequency")
plt.title('Distribution of Reveiw',size = 20)
# Data to plot
labels = data['Type'].value_counts(sort=True).index
sizes = data['Type'].value_counts(sort=True)

colors = ["palegreen", "orangered"]
explode = (0.1, 0)  # explode 1st slice

rcParams['figure.figsize'] = 8, 8
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=270, )

plt.title('Percent of Free App in store', size=20)
plt.show()
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

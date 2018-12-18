import utils
import warnings
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import plotly.offline as py

py.init_notebook_mode(connected=True)
warnings.filterwarnings('ignore')

_, _, data = utils.get_data2()

# Rating distibution
rcParams['figure.figsize'] = 11.7, 8.27
g = sns.kdeplot(data.Rating, color="Red", shade=True)
g.set_xlabel("Rating")
g.set_ylabel("Frequency")
plt.title('Distribution of Rating', size=20)
plt.show()

# Bar chart
g = sns.countplot(x="Category", data=data, palette="Set1")
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
plt.title('Count of app in each category', size=20)
plt.show()

# Pie chart
labels = data['Type'].value_counts(sort=True).index
sizes = data['Type'].value_counts(sort=True)
colors = ["palegreen", "orangered"]
explode = (0.1, 0)  # explode 1st slice
rcParams['figure.figsize'] = 8, 8
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=270, )
plt.title('Percent of Free App in store', size=20)
plt.show()

# Heat map
sns.heatmap(data.corr())
plt.show()

# Box plot
sns.boxplot(x="Category_c", y="Rating", data=data)
plt.show()
sns.boxplot(x="Price", y="Rating", data=data)
plt.show()

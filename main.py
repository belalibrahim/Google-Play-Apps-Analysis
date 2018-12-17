import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import pandas as pd
import numpy as np
import scipy as sc
import random


filename = 'Data/googleplaystore.csv'
predictors, response ,data= utils.get_data(filename)
#sns.pairplot(data, hue="Size", size=3, kind='scatter').add_legend()
sns.heatmap(data.corr())

#plt.imshow(data.corr(), cmap='hot')
plt.show()

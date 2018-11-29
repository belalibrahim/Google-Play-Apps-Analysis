import numpy as np
import pandas as pd


filename = 'Data/googleplaystore.csv'
data = pd.read_csv(filename)

data.Size = data.Size.str.extract('(\d+)')
data.Installs = data.Installs.str.extract('(\d+)')
data.Price = data.Price.str.extract('(\d+)')

data = data.dropna()

data.Size = pd.to_numeric(data.Size)
data.Installs = pd.to_numeric(data.Installs)
data.Price = pd.to_numeric(data.Price)

x = data.loc[:, data.columns != "Rating"]
y = data.Rating

print(data.info())

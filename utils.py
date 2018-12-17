import locale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations


def get_data(filename='Data/googleplaystore.csv'):
    data = pd.read_csv(filename)

    data.Size = data.Size.str.extract('(\d+.\d+|\d+)')
    data.Price = data.Price.str.extract('(\d+.\d+|\d+)')
    data.Installs = data.Installs.str.extract('(\d+(,\d+)*)')

    data = data.dropna()

    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    data.Installs = data.Installs.apply(lambda x: locale.atoi(str(x)))

    data.Size = pd.to_numeric(data.Size)
    data.Price = pd.to_numeric(data.Price)
    data.Installs = pd.to_numeric(data.Installs)
    data.Reviews = pd.to_numeric(data.Reviews)
    data["Last Updated"] = pd.to_datetime(data["Last Updated"])

    predictors = data.loc[:, data.columns != "Rating"]
    response = data.Rating

    return predictors, response, data


# Scaling and cleaning size of installation
def change_size(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x) * 1000000
        return x
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x) * 1000
        return x
    else:
        return None


# Converting Type classification into binary
def type_cat(types):
    if types == 'Free':
        return 0
    else:
        return 1


# Cleaning prices
def price_clean(price):
    if price == '0':
        return 0
    else:
        price = price[1:]
        price = float(price)
        return price


def get_preprocessed_data(filename='Data/googleplaystore.csv'):
    df = pd.read_csv(filename)

    df.dropna(inplace=True)

    # Cleaning Categories into integers
    CategoryString = df["Category"]
    categoryVal = df["Category"].unique()
    categoryValCount = len(categoryVal)
    category_dict = {}

    for i in range(0, categoryValCount):
        category_dict[categoryVal[i]] = i

    df["Category_c"] = df["Category"].map(category_dict).astype(int)

    df["Size"] = df["Size"].map(change_size)

    # Filling Size which had NA
    df.Size.fillna(method='ffill', inplace=True)

    # Cleaning no of installs classification
    df['Installs'] = [int(i[:-1].replace(',', '')) for i in df['Installs']]

    df['Type'] = df['Type'].map(type_cat)

    # Cleaning of content rating classification
    RatingL = df['Content Rating'].unique()
    RatingDict = {}

    for i in range(len(RatingL)):
        RatingDict[RatingL[i]] = i

    df['Content Rating'] = df['Content Rating'].map(RatingDict).astype(int)

    # Dropping of unrelated and unnecessary items
    df.drop(labels=['Last Updated', 'Current Ver', 'Android Ver', 'App'], axis=1, inplace=True)

    # Cleaning of genres
    GenresL = df.Genres.unique()
    GenresDict = {}

    for i in range(len(GenresL)):
        GenresDict[GenresL[i]] = i

    df['Genres_c'] = df['Genres'].map(GenresDict).astype(int)

    df['Price'] = df['Price'].map(price_clean).astype(float)

    # Convert reviews to numeric
    df['Reviews'] = df['Reviews'].astype(int)

    # For dummy variable encoding for Categories
    df_extended = pd.get_dummies(df, columns=['Category'])

    return df, df_extended


def get_data2():

    df, df_extended = get_preprocessed_data()

    X = df.drop(labels=['Category', 'Rating', 'Genres'], axis=1)
    y = df.Rating

    return X, y, df


def subset_selection(x, y):
    x = np.array(x)
    y = np.array(y)

    l = len(x[0])
    best_sub = []
    min_rss = np.inf
    # Iterate over all the features
    for n in range(l + 1):
        # Get all the combinations
        all_comb = list(set(combinations(range(l), n)))
        for i in all_comb:
            x_sub = x[:, list(i)]
            coff = (np.dot(np.dot(np.linalg.inv(np.dot(x_sub.T, x_sub)), x_sub.T), y))
            y_hat = (np.dot(coff.T, x_sub.T))
            rss = sum(np.power(y - y_hat, 2))
            if rss < min_rss:
                min_rss = rss
                best_sub = list(i)
            plt.scatter(n, rss)

    print("Best subset: " + str(best_sub))
    plt.xlabel("Number of features")
    plt.ylabel("RSS")
    plt.title("Subset Selection")
    plt.show()

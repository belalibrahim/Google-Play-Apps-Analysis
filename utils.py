import locale
import pandas as pd


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
# scaling and cleaning size of installation
def change_size(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x) * 1000000
        return (x)
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x) * 1000
        return (x)
    else:
        return None
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



def get_data_2(filename='Data/googleplaystore.csv'):
    df = pd.read_csv(filename)
    #df.info()
    df.dropna(inplace=True)
    #df.info()
    #df.head()
    # Cleaning Categories into integers
    CategoryString = df["Category"]
    categoryVal = df["Category"].unique()
    categoryValCount = len(categoryVal)
    category_dict = {}
    for i in range(0, categoryValCount):
        category_dict[categoryVal[i]] = i
    df["Category_c"] = df["Category"].map(category_dict).astype(int)
    df["Size"] = df["Size"].map(change_size)
    # filling Size which had NA
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
    # dropping of unrelated and unnecessary items
    df.drop(labels=['Last Updated', 'Current Ver', 'Android Ver', 'App'], axis=1, inplace=True)
    # Cleaning of genres
    GenresL = df.Genres.unique()
    GenresDict = {}
    for i in range(len(GenresL)):
        GenresDict[GenresL[i]] = i
    df['Genres_c'] = df['Genres'].map(GenresDict).astype(int)
    df['Price'] = df['Price'].map(price_clean).astype(float)
    # convert reviews to numeric
    df['Reviews'] = df['Reviews'].astype(int)
    #df.info()
    #df.head()
    # for dummy variable encoding for Categories
    df2 = pd.get_dummies(df, columns=['Category'])
    #df2.head()
    return df2,df






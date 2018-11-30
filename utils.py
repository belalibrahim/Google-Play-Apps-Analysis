import pandas as pd
import locale


def get_data(filename):
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

    return predictors, response

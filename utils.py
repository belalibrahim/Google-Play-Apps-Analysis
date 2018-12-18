import locale
import models
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


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

    # TODO For regularization
    df = df[:700]

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
            coeff = (np.dot(np.dot(np.linalg.inv(np.dot(x_sub.T, x_sub)), x_sub.T), y))
            y_hat = (np.dot(coeff.T, x_sub.T))
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


def apply_reg(x, y):
    x = np.array(x)
    y = np.array(y)

    x_c = np.zeros(list(x.shape))
    n_features = x.shape[1]

    for i in range(n_features):
        x_c[:, i] = (x[:, i] - np.mean(x[:, i])) / np.std(x[:, i])

    lambda_ = np.arange(1000, 0, -1)
    coefficients = np.zeros([n_features, 1000])
    df_lambda = list()

    for i in range(len(lambda_)):
        coefficients[:, i] = (
            np.dot(np.dot(np.linalg.inv(np.dot(x_c.T, x_c) + (np.dot(lambda_[i], np.eye(n_features)))), x_c.T), y))
        df_lambda.append(
            np.trace(np.dot(x_c, np.dot(np.linalg.inv(np.dot(x_c.T, x_c) + np.dot(lambda_[i], np.eye(n_features))), x_c.T))))

    for i in range(n_features):
        plt.plot(df_lambda, coefficients[i, :])

    plt.xlabel("df(λ)")
    plt.ylabel("Coefficients")
    plt.title("Regularization")
    plt.show()

    n_splits = 5
    kf = KFold(n_splits=n_splits)

    lambda_ = np.arange(5, 0, -0.05)
    all_tr_err = list()
    all_tst_err = list()
    for i in range(len(lambda_)):
        rss = 0
        rss_ts = 0
        for train_index, test_index in kf.split(x_c):
            X_train, X_test = x_c[train_index], x_c[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train_hat = np.mean(y_train) + np.dot(coefficients[:, i], X_train.T)
            y_test_hat = np.mean(y_train) + np.dot(coefficients[:, i], X_test.T)
            rss += np.mean((y_train - y_train_hat) ** 2) + (lambda_[i] * np.mean((coefficients[:, i] ** 2)))
            rss_ts += np.mean((y_test - y_test_hat) ** 2)
        all_tr_err.append(rss / n_splits)
        all_tst_err.append(rss_ts / n_splits)

    plt.plot(np.log(lambda_), all_tst_err)
    plt.show()


def apply_over(x, y):

    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    x_new = np.column_stack((x, np.power(x, 2), np.power(x, 3)))
    print(x_new.shape)

    x_c = np.zeros(list(x_new.shape))
    n_features = x_new.shape[1]

    for i in range(n_features):
        x_c[:, i] = (x_new[:, i] - np.mean(x_new[:, i])) / np.std(x_new[:, i])

    lambda_ = np.arange(1000, 0, -1)
    coefficients = np.zeros([n_features, 1000])
    df_lambda = list()

    for i in range(len(lambda_)):
        coefficients[:, i] = (
            np.dot(np.dot(np.linalg.inv(np.dot(x_c.T, x_c) + (np.dot(lambda_[i], np.eye(n_features)))), x_c.T), y))
        df_lambda.append(
            np.trace(
                np.dot(x_c, np.dot(np.linalg.inv(np.dot(x_c.T, x_c) + np.dot(lambda_[i], np.eye(n_features))), x_c.T))))


    n_splits = 5
    kf = KFold(n_splits=n_splits)

    lambda_ = np.arange(5, 0, -0.05)
    all_tr_err = list()
    all_tst_err = list()
    for i in range(len(lambda_)):
        rss = 0
        rss_ts = 0
        for train_index, test_index in kf.split(x_c):
            X_train, X_test = x_c[train_index], x_c[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train_hat = np.mean(y_train) + np.dot(coefficients[:, i], X_train.T)
            y_test_hat = np.mean(y_train) + np.dot(coefficients[:, i], X_test.T)
            rss += np.mean((y_train - y_train_hat) ** 2) + (lambda_[i] * np.mean((coefficients[:, i] ** 2)))
            rss_ts += np.mean((y_test - y_test_hat) ** 2)
        all_tr_err.append(rss / n_splits)
        all_tst_err.append(rss_ts / n_splits)

    plt.plot(np.log(lambda_), all_tst_err)
    plt.show()


def Evaluationmatrix(y_true, y_predict, model):
    print('Mean Squared Error ' + str(model) + ': ' + str(metrics.mean_squared_error(y_true, y_predict)))
    print('Mean absolute Error ' + str(model) + ': ' + str(metrics.mean_absolute_error(y_true, y_predict)))
    print('Mean squared Log Error ' + str(model) + ': ' + str(metrics.mean_squared_log_error(y_true, y_predict)))
    print('\n')


def models_compare(x, y):

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

    results_svm = models.SVM(X_train, y_train, X_test)
    sns.regplot(results_svm, y_test, color='red', label='SVM')
    Evaluationmatrix(y_test, results_svm, "SVM")

    results_tree = models.TREE(X_train, y_train, X_test)
    sns.regplot(results_tree, y_test, color='green', label='TREE')
    Evaluationmatrix(y_test, results_tree, "TREE")

    results_ridge = models.RIDGE(X_train, y_train, X_test)
    sns.regplot(results_ridge, y_test, color='orange', label='RIDGE')
    Evaluationmatrix(y_test, results_ridge, "RIDGE")

    results_knn = models.KNN(X_train, y_train, X_test)
    sns.regplot(results_knn, y_test, color='yellow', label='KNN')
    Evaluationmatrix(y_test, results_knn, "KNN")

    results_lr = models.LR(X_train, y_train, X_test)
    sns.regplot(results_lr, y_test, color='blue', label='LR')
    Evaluationmatrix(y_test, results_lr, "LR")

    results_rfr = models.RFR(X_train, y_train, X_test)
    sns.regplot(results_rfr, y_test, color='black', label='RFR')
    Evaluationmatrix(y_test, results_rfr, "RFR")

    plt.title('Models Comparison')
    plt.xlabel('Predicted Ratings')
    plt.ylabel('Actual Ratings')
    plt.legend()
    plt.show()

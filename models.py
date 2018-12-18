from sklearn import svm
from sklearn import tree
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def SVM(X_train, y_train, X_test):
    model = svm.SVR()
    model.fit(X_train, y_train)
    results = model.predict(X_test)
    return results


def TREE(X_train, y_train, X_test):
    model = tree.DecisionTreeRegressor()
    model.fit(X_train, y_train)
    results = model.predict(X_test)
    return results


def RIDGE(X_train, y_train, X_test):
    model = Ridge()
    model.fit(X_train, y_train)
    results = model.predict(X_test)
    return results


def KNN(X_train, y_train, X_test):
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    results = model.predict(X_test)
    return results


def LR(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    results = model.predict(X_test)
    return results


def RFR(X_train, y_train, X_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    results = model.predict(X_test)
    return results
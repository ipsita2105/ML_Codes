#!/usr/bin/env python3

import sys
import re
import os
import math
from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd

import sklearn
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

import matplotlib.pylab as plt
import seaborn

def main():
    
    f = open("PDBbind2015_refined-core.dat")
    line = f.readline()

    x = []
    y = []

    datalist = []

    while 1:

        line = f.readline().strip("\n").split(" ")

        if line == ['']:
            break
        y.append(float(line[0]))
        x.append(float(line[1]))
        
        datalist.append([float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6])])


    data = np.array(datalist)
    #sdata = preprocessing.scale(data)
    data = preprocessing.normalize(data)

    X = data
    Y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

    # Fit regression models

    # Linear Regression
    m_lr = linear_model.LinearRegression()
    kfold = KFold(n_splits=10, random_state=21)
    cv_result = cross_val_score(m_lr, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    print('Linear Regression CVE = ',cv_result.mean(),',',cv_result.std())


    # Ridge Regression
    m_rg = linear_model.Ridge()
    kfold = KFold(n_splits=10, random_state=21)
    cv_result = cross_val_score(m_rg, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    print('Ridge Regression CVE = ',cv_result.mean(),',',cv_result.std())
    
    # Lasso 
    m_lasso = linear_model.Lasso()
    kfold = KFold(n_splits=10, random_state=21)
    cv_result = cross_val_score(m_lasso, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    print('Lasso Regression CVE = ',cv_result.mean(),',',cv_result.std())   

    # Elastic Net
    m_net = linear_model.ElasticNet()
    kfold = KFold(n_splits=10, random_state=21)
    cv_result = cross_val_score(m_net, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    print('Elastic Net CVE = ',cv_result.mean(),',',cv_result.std())

    # Polynomial Fit
    svr_poly = SVR(kernel='poly'  , C=1e3, degree=2)
    kfold = KFold(n_splits=10, random_state=21)
    cv_result = cross_val_score(svr_poly, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    print('Poly CVE = ',cv_result.mean(),',',cv_result.std())


    # Exponential Fit
    svr_rbf  = SVR(kernel='rbf'   , C=1e3, gamma=0.1)
    kfold = KFold(n_splits=10, random_state=21)
    cv_result = cross_val_score(svr_rbf, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    print('RBF CVE = ',cv_result.mean(),',',cv_result.std())

    #for correlation plot
    x_train = pd.DataFrame(x_train)
    corr_df = x_train.corr(method='pearson')

    mask = np.zeros_like(corr_df)
    mask[np.triu_indices_from(mask)] = True

    seaborn.heatmap(corr_df, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0, mask=mask, linewidths=2.5)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()


if __name__ == '__main__':
    main()
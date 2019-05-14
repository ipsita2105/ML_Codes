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
import matplotlib.pylab as plt
from sklearn.metrics import r2_score, mean_squared_error
import seaborn

def correlation(x, y):

        N      = len(x)
        sumxy  = 0.0
        sumx   = 0.0
        sumy   = 0.0
        sumxsq = 0.0
        sumysq = 0.0

        for i in range(0, N):
            sumxy = sumxy + x[i]*y[i]
            sumx  = sumx  + x[i]
            sumy  = sumy  + y[i]
            sumxsq = sumxsq + x[i]**2
            sumysq = sumysq + y[i]**2

        numer = N*(sumxy)-(sumx*sumy)
        denom = math.sqrt(((N*sumxsq)-(sumx**2))*((N*sumysq) - (sumy**2)))

        return numer/denom

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

    X = pd.DataFrame(data)
    Y = pd.DataFrame(y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
  

    model = linear_model.LinearRegression(fit_intercept=True, normalize=True,copy_X=True)
    model.fit(x_train, y_train)

    # Testing part
    tfile = open("PDBbind2015_core.dat")
    line = tfile.readline()

    x = []
    y = []

    datalist = []

    while 1:

        line = tfile.readline().strip("\n").split(" ")

        if line == ['']:
            break
        y.append(float(line[0]))
        x.append(float(line[1]))
        
        datalist.append([float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6])])
        
    data = np.array(datalist)

    X = pd.DataFrame(data)
    Y = pd.DataFrame(y)

    y_pred = model.predict(X)
    p = pearsonr(y_pred, Y)

    print('Mean squared error: %.2f' % mean_squared_error(Y, y_pred))
    print('Correlation:','r =',p[0],'p value =',p[1])


if __name__ == '__main__':
    main()

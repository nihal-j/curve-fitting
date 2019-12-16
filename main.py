import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import preprocessing
from gradient_descent import Model

def plot_loss(errors, numIters, interval):

    '''
    '''
    
    xLabels = [i for i in range(0, numIters) if i % interval == 0]
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.plot(xLabels, [errors[i] for i in range(len(errors)) if i % interval == 0])
    plt.show()

if __name__ == '__main__':

    dataDf = pd.read_csv('data.csv')
    x = np.array(dataDf['LATITUDE'].tolist())
    y = np.array(dataDf['LONGITUDE'].tolist())
    T = np.array(dataDf['ALTITUDE'].tolist())

    normalizedx = preprocessing.normalize(x)
    normalizedy = preprocessing.normalize(y)
    normalizedT = preprocessing.normalize(T)

    x_train, y_train, T_train, x_val, y_val, T_val, x_test, y_test, T_test = preprocessing.split(normalizedx, normalizedy, T)
    trainCount = len(T_train)

    featureMatrix, _ = preprocessing.generate_features(x_train, y_train, trainCount, 1)

    numIters = 1000
    eta = 0.000001
    lamb = 0
    deg = 1
    reg = 'L2'

    linearModel = Model(np.array([]), featureMatrix, T_train, deg, numIters, eta, lamb, reg)
    errors = linearModel.generate_model()
    # errors = linearModel.generate_stochastic_model()
    # linearModel.generate_normal_equation_model()

    X_val, _ = preprocessing.generate_features(x_val, y_val, len(x_val), 1)


    H = linearModel.predict(X_val, linearModel.W)
    print('R2 error: ', linearModel.calc_R2(T_val, H)) 
    print('RMS error: ', linearModel.calc_rmse(T_val, H))
    print(linearModel.W)

    plot_loss(errors, numIters, 20)
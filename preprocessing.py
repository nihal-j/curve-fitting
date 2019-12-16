import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def normalize(vec, type='standardization'):
    
    '''
    Performs normalization on the vector. The type of normalization can be standardization or
    min-max normalization.
    
    Agruments:
        vec: the vector that has to be normalized
        type: 'standardization' or 'min-max'. Default is standardization.
        
    Returns:
        vec: the normalized vector.
    '''
    
    if type == 'standardization':
        return (vec - np.mean(vec)) / np.std(vec)
    
    if type == 'min-max':
        return (vec - min(vec)) / (max(vec) - min(vec))

def split(x, y, T):

    '''
    Splits the vectors into training, cross-validation and test sets,
    following 80%, 10% and 10% sizes respectively.
    
    Arguments:
        x: vector x of data (latitude).
        y: vector y of data (longitude).
        T: target variable vector of data (altitude).
        
    Returns:
        x_train, y_train, T_train, x_val, y_val, T_val, x_test, y_test, T_test
        following the description above.
    '''
    
    X = [(x[i], y[i]) for i in range(len(x))]
    X_train, X_test, T_train, T_test = train_test_split(X, T, test_size = 0.2, random_state = 0)
    X_test, X_val, T_test, T_val = train_test_split(X_test, T_test, test_size = 0.5, random_state = 42)
    
    x_test = np.array([X_test[i][0] for i in range(len(X_test))])
    y_test = np.array([X_test[i][1] for i in range(len(X_test))])
    
    x_val = np.array([X_val[i][0] for i in range(len(X_val))])
    y_val = np.array([X_val[i][1] for i in range(len(X_val))])
    
    x_train = np.array([X_train[i][0] for i in range(len(X_train))])
    y_train = np.array([X_train[i][1] for i in range(len(X_train))])
                    
    return x_train, y_train, T_train, x_val, y_val, T_val, x_test, y_test, T_test

def generate_features(x, y, N, deg):

    '''
    Generates the feature matrix for each pow of y (dy), get all pow of x (dx), 
    such that dx + dy = deg
    
    The feature matrix looks as:
    
    [[1, x1, x1^2, y1, x1y1, y1^2]
    [1, x2, x2^2, y2, x2y2, y2^2]
    .  .
    .  .
    [1, xN, xN^2, yN, xNyN, yN^2]]
    
    Arguments:
        x: vector x of data (latitude)
        y: vector y of data (longitude)
        N: no. of training examples
        deg: maximum degree upto which features need to be calculated
        
    Returns:
        featureMatrix: the featureMatrix as described
        d: no. of features (no. of columns in featureMatrix)
    '''
    
    featureMatrix = []
    
    # number of features
    d = 0
    
    if N == 1:
        x = [x]
        y = [y]
    
    for n in range(N):
        row = []
        for i in range(deg + 1):
            for j in range(deg - i + 1):
                term = (x[n]**j) * (y[n]**i)
                row.append(term)
                if n == 0:
                    d += 1
        featureMatrix.append(row)
    
    # converting to a numpy array
    featureMatrix = np.array(featureMatrix)
    
    return featureMatrix, d


if __name__ == '__main__':

    dataDf = pd.read_csv('data.csv')
    x = np.array(dataDf['LATITUDE'].tolist())
    y = np.array(dataDf['LONGITUDE'].tolist())
    T = np.array(dataDf['ALTITUDE'].tolist())

    normalizedx = normalize(x)
    normalizedy = normalize(y)
    normalizedT = normalize(T)

    x_train, y_train, T_train, x_val, y_val, T_val, x_test, y_test, T_test = split(normalizedx, normalizedy, T)
import random

import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

import preprocessing

class Model:

    def __init__(self, W, X, T, deg, maxIter, eta, lamb=0, reg='L2'):

        self.W = W
        self.X = X
        self.T = T
        self.deg = deg
        self.maxIter = maxIter
        self.eta = eta
        self.lamb = lamb
        self.reg = reg

    def update_weights(self):
    
        '''
        Updates weights using regularization (if lamb != 0), following
        'L1' or 'L2' regularization. Default is 'L2' regularization.     
        '''

        H = self.X.dot(self.W)

        features = np.transpose(self.X)
        error = H - self.T
        
        # print(features.reshape((len(features),1)))
        # print(error.reshape((1,1)))
        delta = features.dot(error)
        if self.reg == 'L2':
            # print(self.lamb)
            # print(delta.shape)
            # print(self.W.shape)
            delta += np.multiply(self.lamb, self.W)
        else:
            Wdel = np.copy(self.W)
            for i in range(len(Wdel)):
                if Wdel[i] > 0:
                    Wdel[i] = 1
                elif Wdel[i] < 0:
                    Wdel[i] = -1
                else:
                    Wdel[i] = 0
            delta += self.lamb * Wdel
        
        self.W = self.W - (self.eta * delta)

    def predict(self, X, W):
        
        '''
        Performs prediction, given the weights and features using
        the formula: Y = X.theta.

        Arguments:
            X:  feature matrix.
            W:  weights.

        Returns:
            H: the prediction values.
        '''

        H = X.dot(W)
        return H

    def generate_model(self):

        '''
        Performs gradient descent on this object's attributes.
        
        Hyperparameter tuning can be performed by changing values of lambda (for regularization),
        eta (for optimizing learning rate) and degree (for finding the threshold for overfitting)
        during initialization of this object.

        If pre-trained weights are not used for initialization, then random initialization is done
        for the weights.

        The termination condition is specified using max number of iterations.

        This method does not return the learned weights but only updates the class variables and
        they can be accessed using the object's instance.
        The loss over the training iterations is returned as a Python list for analysis.
        '''

        numOfFeatures = self.X.shape[1]
        if not np.size(self.W):
            # initial weights vector
            # random initialization
            random.seed(12)
            self.W = np.array([random.random() for i in range(numOfFeatures)])
            # zero initialization
            # self.W = np.array([0 for i in range(d)])
        
        prevError = 0
        currentError = 0
        errors = []
        
        for i in tqdm(range(self.maxIter)):

            H = self.X.dot(self.W)
            if self.reg == 'L2':
                E = (0.5*(H - self.T).dot(np.transpose(H - self.T))) + (0.5*self.lamb*np.sum(self.W*self.W))
            else:
                E = (0.5*(H - self.T).dot(np.transpose(H - self.T))) + (0.5*self.lamb*sum(np.abs(self.W)))
            prevError = currentError
            currentError = E
            errors.append(currentError)
            self.update_weights()
        
        return errors

    def generate_stochastic_model(self):

        '''
        Performs stochastic gradient descent on this object's attributes.
        For more details, refer documentation of generate_model() function.

        Number of iterations should be tuned to specify number of epochs.
        '''

        featureMatrix = np.copy(self.X)
        targetValues = np.copy(self.T)
        N = len(featureMatrix)
        errors = []

        numOfFeatures = self.X.shape[1]
        if not np.size(self.W):
            # initial weights vector
            # random initialization
            random.seed(12)
            self.W = np.array([random.random() for i in range(numOfFeatures)])
            # zero initialization
            # self.W = np.array([0 for i in range(d)])

        for i in tqdm(range(self.maxIter)):
            
            self.X = featureMatrix[i%N]
            self.T = np.array([targetValues[i%N]])
            H = np.array([self.X.dot(self.W)])
            # print(self.T.shape)
            # print(H)
            E = (0.5*(H - self.T).dot(np.transpose(H - self.T))) + (0.5*self.lamb*np.sum(self.W * self.W))

            errors.append(E)
            
            self.update_weights()
            
        return errors

    def generate_normal_equation_model(self):

        '''
        Generates an estimate for model parameters using an analytic method.
        The equation used is:
            theta = ((inv(X_trans.X)).(X_trans)).T
        
        No errors are returned because there is no training involved.
        '''

        self.W = np.matmul(np.transpose(self.X), self.X)
        self.W = np.linalg.inv(self.W)
        self.W = np.matmul(self.W, np.transpose(self.X))
        self.W = np.dot(self.W, self.T)
        
        return self.W

    def calc_R2(self, T, H):
    
        '''
        Calculates the R^2 score for this object's model.
        Refer https://en.wikipedia.org/wiki/Coefficient_of_determination for more details.

        Arguments:
            T:  the truth values for the target variable.
            H:  the predicted values for the target variable.

        Returns:
            r2: the R^2 score.
        '''

        tss = np.sum((T - np.mean(T))*(T - np.mean(T)))
        rss = np.sum((T - H)*(T - H))
        r2 = 1 - (rss/tss)
        
        return r2

    def calc_rmse(self, T, H):

        '''
        Calculates the RMSE score for this object's model.
        Refer https://en.wikipedia.org/wiki/Root-mean-square_deviation for more details.

        Arguments:
            T:  the truth values for the target variable.
            H:  the predicted values for the target variable.

        Returns:
            r2: the R^2 score.
        ''' 

        se = np.sum((T - H)*(T - H))
        mse = se/len(T)
        rmse = mse ** 0.5
        
        return rmse

if __name__ == '__main__':

    dataDf = pd.read_csv('data.csv')
    x = np.array(dataDf['LATITUDE'].tolist())
    y = np.array(dataDf['LONGITUDE'].tolist())
    T = np.array(dataDf['ALTITUDE'].tolist())

    normalizedx = preprocessing.normalize(x)
    normalizedy = preprocessing.normalize(y)
    normalizedT = preprocessing.normalize(T)

    x_train, y_train, T_train, x_val, y_val, T_val, x_test, y_test, T_test = preprocessing.split(normalizedx, normalizedy, T)
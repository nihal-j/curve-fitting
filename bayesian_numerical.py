import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from tqdm import tqdm
from scipy.interpolate import make_interp_spline, BSpline
import math

def binomial_sample(n, p):

    '''
    Returns a sample of a fixed proportion of successes, assuming binomial
    distribution.

    Arguments:
        n: no. of trials
        p: probability of success

    Returns:
        observations: list indicating the results of trails maintaing n*p proportion
    '''

    heads = int(p*n)
    observations = np.array([1]*heads + [0]*(n - heads))
    np.random.shuffle(observations)
    
    return observations

def gamma(x):

    '''Calculates gamma function value for x using library function.'''

    return scipy.special.gamma(x)

def beta(x, a, b):
    
    '''Calculates beta distribution probability value for x using a and b parameter values.'''

    factor = gamma(a+b) / (gamma(a) * gamma(b))
    return factor * (x**(a-1)) * ((1-x)**(b-1))

def bernoulli(x, mu):

    '''Calculates bernoulli distribution probability value for x using mu parameter value.''''

    return (mu**(x)) * ((1-mu)**(1-x))

def binomial(x, n, mu):

    '''Calculates binomial distribution probability value for x using n, mu paramter values.'''

    factor = math.factorial(n) // math.factorial(x) // math.factorial(n - x)
    return factor * (mu**x) * ((1 - mu)**(n-x))

def estimate_seq(observations):

    '''
    Estimates the beta distribution sequentially, i.e., the posterior is updated one observation at a time.
    
    An initial beta distribution is assumed such that its mean is in range [0.4,0.6].
    
    It can be shown that the posterior for a binomial distribution (in this case head tossing) has a conjugate beta prior distribution.
    With each observation made, the posterior is obtained numerically using Bayes' Theorem:
        P (mu | x) = P(x | mu).P(mu)/P(x),
        where,
        P(x | mu) is the binomial distribution probability for making observations,
        P(x) can be estimated using integration of the numerator factor (also done numerically).

    Arguments:
        observations:   list of obeservations. Each observation is either 0 or 1.
                        0 indicates a tail outcome; 1 indicates a head outcome.

    Returns:
        a:  parameter a for beta distribution.
        b:  parameter b for beta distribution.
    '''

    # initializing the beta distribution such that mean = 0.4
    a = 4
    b = 6

    mu = np.random.random(10000)
    mu = np.array(sorted(mu))
    prior = beta(mu, a, b)
    posterior = prior

    for observation in observations:

        likelihood = bernoulli(observation, mu)
        posterior = prior*likelihood
        
        integral = 0
        for i in range(len(posterior)):
            if i > 0:
                integral += posterior[i]*(mu[i]-mu[i-1])
        posterior /= integral

        prior = posterior

    return mu, posterior

def estimate_batch(observations):

    '''
    Estimates the beta distribution in one go, i.e., the posterior is updated by considering the entire set of observations simultaneously.
    
    An initial beta distribution is assumed such that its mean is in range [0.4,0.6].
    
    It can be shown that the posterior for a binomial distribution (in this case head tossing) has a conjugate beta prior distribution.
    With each observation made, the posterior is obtained numerically using Bayes' Theorem:
        P (mu | x) = P(x | mu).P(mu)/P(x),
        where,
        P(x | mu) is the binomial distribution probability for making observations,
        P(x) can be estimated using integration of the numerator factor (also done numerically).

    Arguments:
        observations:   list of obeservations. Each observation is either 0 or 1.
                        0 indicates a tail outcome; 1 indicates a head outcome.

    Returns:
        a:  parameter a for beta distribution.
        b:  parameter b for beta distribution.
    '''

    a = 4
    b = 6

    mu = np.random.random(10000)
    mu = np.array(sorted(mu))
    n = len(observations)
    x = sum(observations)

    likelihood = binomial(x, n, mu)
    prior = beta(mu, a, b)
    posterior = likelihood * prior

    integral = 0
    for i in range(len(posterior)):
        if i > 0:
            integral += posterior[i]*(mu[i]-mu[i-1])
    posterior /= integral

    return mu, posterior


if __name__ == '__main__':

    '''Driver function.'''

    distribution1 = estimate_seq(binomial_sample(160, 0.9))
    distribution2 = estimate_batch(binomial_sample(160, 0.9))
    # plt.scatter(distribution[0], distribution[1])
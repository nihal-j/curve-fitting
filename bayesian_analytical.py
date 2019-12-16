import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from tqdm import tqdm_notebook, tqdm
from scipy.interpolate import make_interp_spline, BSpline

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

def update(a, b, currentObservation):

    '''
    Updates the probability distribution to obtain the posterior by accounting 
    for the currentObservation.

    Arguments:
        a:  parameter a for prior beta distribution.
        b:  parameter b for prior beta distribution.
        currentObservation: value of currentObservation.

    Returns:
        a:  parameter a for posterior beta distribution.
        b:  parameter b for posterior beta distribution.
    '''

    if currentObservation == 1:
        a += 1
    else:
        b += 1

    return a, b

def gamma(x):

    '''Calculates gamma function value for x using library function.'''

    return scipy.special.gamma(x)

def plot_beta(a, b, n, j):

    '''
    Utility function to plot the beta distribution, given its parameters, a and b.

    Arguments:
        a:  parameter a for beta distriubtion.
        b:  parameter b for beta distribution.
        n:  number of points that are to be plotted. Greater n => greater smoothness.
        j:  serial number to index the plot.
    '''

    x = np.random.random(n)
    y = np.zeros(n)

    tmp = sorted(zip(x,y), key = lambda x: x[0])
    for i in range(len(tmp)):
        x[i] = tmp[i][0]
        y[i] = tmp[i][1]

    for i in range(n):
        factor = gamma(a+b)/(gamma(a)*gamma(b))
        y[i] = factor * (x[i]**(a-1)) * ((1 - x[i])**(b-1))

    plt.clf()
    plt.cla()
    plt.close()

    xnew = np.linspace(x.min(), x.max(), 1000)
    spline = make_interp_spline(x, y, k = 3)
    ysmooth = spline(xnew)

    plt.plot(xnew, ysmooth)
    
    ax = plt.gca()
    ax.annotate('Iteration '+str(j)+'\nmu = '+str(x[y.argmax()])[:5], xy=(x.max(),y.max()), xytext=(0,12))
    plt.xlabel('mu (Mean of bernoulli trial)')
    plt.ylabel('beta(mu)')
    plt.yticks(np.arange(0, 17, 1))
    plt.xticks(np.arange(1.1, step=0.1))
    plt.show()
    # plt.savefig('fig/' + format(j, '03d'))
    # plt.savefig('0.5.png')

def estimate_p_seq(observations):

    '''
    Estimates the beta distribution sequentially, i.e., the posterior is updated one observation at a time.
    
    An initial beta distribution is assumed such that its mean is in range [0.4,0.6].
    
    It can be shown that the posterior for a binomial distribution (in this case head tossing) has a conjugate beta prior distribution.
    With each observation made, the posterior is obtained as follows:
        a' <- a + x,
        b' <- b + x,
    where x is the value of the observation in {0, 1}.

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
    plot_beta(a, b, 1000, 0)

    for i, currentObservation in tqdm(enumerate(observations)):
        print(i)
        a, b = update(a, b, currentObservation)
        plot_beta(a, b, 1000, i+1)

    return a, b

def estimate_p_batch(observations):

    '''
    Estimates the beta distribution in one go, i.e., the posterior is updated by considering the entire set of observations simultaneously.
    
    An initial beta distribution is assumed such that its mean is in range [0.4,0.6].
    
    It can be shown that the posterior for a binomial distribution (in this case head tossing) has a conjugate beta prior distribution.
    With each observation made, the posterior is obtained as follows:
        a' <- a + m,
        b' <- b + l, 
        where,
        m is the number of 1's (head observations),
        n is the number of 0's (tail observations).

    Arguments:
        observations:   list of obeservations. Each observation is either 0 or 1.
                        0 indicates a tail outcome; 1 indicates a head outcome.

    Returns:
        a:  parameter a for beta distribution.
        b:  parameter b for beta distribution.
    '''

    a = 4
    b = 6
    plot_beta(a, b, 1000, 161)

    m = sum(observations)
    l = len(observations) - m

    a += m
    b += l

    plot_beta(a, b, 1000, 162)

    return a, b

if __name__ == '__main__':

    '''Driver function.'''

    # print(binomial_sample(160, 0.25))
    a, b = estimate_p_seq(binomial_sample(160, 0.9))
    print(a,b)
    a, b = estimate_p_batch(binomial_sample(160, 0.9))
    print(a,b)
    # plot_beta(25, 25, 1000, 0)
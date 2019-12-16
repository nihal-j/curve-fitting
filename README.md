# Curve Fitting
This repository contains an implementation of gradient descent to perform linear regression as a solution to the problem of curve fitting. 

The dataset used for the demonstration of the algorithm can be found here:
https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+%28North+Jutland%2C+Denmark%29.

The input data contains elevation information to a 2D road network in *North Jutland, Denmark*. It
contains the following attributes:
1. **OSM_ID**:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;OpenStreetMap ID for each road segment or edge in the graph.
2. **LONGITUDE**:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Google format) longitude
3. **LATITUDE**:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Google format) latitude
4. **ALTITUDE**:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Height in meters.

The dataset contains 434874 instances.

For the purpose of curve fitting, we make use of **LONGITUDE** and **LATITUDE** to estimate the target variable **ALTITUDE**.
<br><br><br>

# Bayesian Estimation
The purpose of this segment of the repository is to demonstrate the use of Bayesian methods to estimate population parameters for suggesting a probability distribution to fit the given set of observations.

To this end, we make use of the classic head-and-tail series of experiments. We know that this has an underlying Binomial distribution. We draw random samples from a Binomial distribution with parameters p = 0.9, n = 160.

We then make use of Bayes' Theorem and the fact that the conjugate prior of a Binomial distribution is a Beta distribution to estimate p and n given only the set of observations.

This estimation is done using both numerical and analytical methods.

Considering one observation at a time, also known as sequential learning, we were able to obtain the following distribution at the end of  n observations:

![Sequential Learning GIF](https://github.com/nihal-j/curve-fitting/blob/master/images/animation.gif "Sequential Learning Using Bayesian Methods")
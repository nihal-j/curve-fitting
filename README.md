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

For the purpose of curve fitting, we make use of **LONGITUDE** and **LATITUDE** to estimate the **target** variable **ALTITUDE**.
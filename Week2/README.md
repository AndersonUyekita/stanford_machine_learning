# Standford Machine Learning - Week 02

#### Tags
* Author : AH Uyekita
* Title  :  _Linear Regression with One Variable_
* Date   : 24/02/2019
* Course : Machine Learning
    * **Instructor:** Andrew Ng

***

## Multivariate Linear Regression

The Multivariate Linear Regression is a generalization of the Univariate Linear Regression. The Equation (1) shows it.

<p align="center"><img src="/Week2/tex/e83e4eea417923e1961c87e4fec760ee.svg?invert_in_darkmode&sanitize=true" align=middle width=362.39509845pt height=35.671178399999995pt/></p>

Where:

* <img src="/Week2/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/>: It is the observations value, and;
* <img src="/Week2/tex/455b7e5df6537b98819492ec6537494c.svg?invert_in_darkmode&sanitize=true" align=middle width=13.82140154999999pt height=22.831056599999986pt/>: The feature/variable parameter;
* <img src="/Week2/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/>: Number of features, and;
* <img src="/Week2/tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/>: Number of observations.

For convenience of notation, define <img src="/Week2/tex/e714a3139958da04b41e3e607a544455.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/> equal to 1. Converting equation (1) in matrix notation.

<p align="center"><img src="/Week2/tex/8522983ef70fea3deea32b2a7dba5aae.svg?invert_in_darkmode&sanitize=true" align=middle width=285.62745915pt height=88.76800184999999pt/></p>

Where:

* <img src="/Week2/tex/a4ca9bf3d588ae6d323a340248299dc4.svg?invert_in_darkmode&sanitize=true" align=middle width=17.70724559999999pt height=27.6567522pt/>: Row vector of parameters, and;
* <img src="/Week2/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/>: Column vector values.

### Gradient Descent in Multivariate Linear Regression

The process is almost the same, which differ is the quantity of the features. The Equation (3) shows the generic Cost Function.

<p align="center"><img src="/Week2/tex/67d66dddf34d4e0c0cc20666c821ca91.svg?invert_in_darkmode&sanitize=true" align=middle width=225.31515929999998pt height=44.89738935pt/></p>

Updating the parameters.

<p align="center"><img src="/Week2/tex/b7d4118bfda4a25c31c4aec1ab8e74e8.svg?invert_in_darkmode&sanitize=true" align=middle width=283.64357175pt height=44.89738935pt/></p>

#### Feature Scaling

In some cases it is necessary to scale the features because the difference between ranges (of the features) could leverage the Gradient Descent convergence. It means if one feature has range from -3 to +3 and the other from -10.000 to +10.000, the difference is too high and the convergence process will last too much.

A way to understand it, is plotting a contours plot using the parameters as axis x and y, and the third axis as Cost Function. Figure 1 shows an example.

![Figure 1 - Contours Plot.](01-img/ml_week_02_01.png)

<center><em>Figure 1 - Contours Plot of Cost Function.</em></center><br>

Due to the skewness of this plot, the Gradient Descent will take several steps (iterations) doing a zig-zag by reach the center of the figure (where we supose it is the minimum value of <img src="/Week2/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/>).

On the other hand, If you performs the feature scaling, Figure 1 will change to seems like Figure 2.

![Figure 2 - Contours Plot Scaled.](01-img/ml_week_02_02.png)

<center><em>Figure 2 - Contours Plot Scaled.</em></center><br>

As a result of the feature scaling, the convergence process in much faster.

The simplest way to do it is substracting the average and dividing it by the range, as shown in equation (5).

<p align="center"><img src="/Week2/tex/4f1d62c5129103569f61ca8db3a12162.svg?invert_in_darkmode&sanitize=true" align=middle width=192.49211354999997pt height=37.9216761pt/></p>

Where:

* <img src="/Week2/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/>: Feature <img src="/Week2/tex/3def24cf259215eefdd43e76525fb473.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32504519999999pt height=27.91243950000002pt/>;
* <img src="/Week2/tex/b3b70cf643b09ec0ba9f3fe61dfc3040.svg?invert_in_darkmode&sanitize=true" align=middle width=85.20095595pt height=24.65753399999998pt/>: Average of all values of <img src="/Week2/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/>, and;
* <img src="/Week2/tex/4f0ad1257b412cb14381be7758045979.svg?invert_in_darkmode&sanitize=true" align=middle width=42.51348089999998pt height=14.15524440000002pt/>: The subtraction of <img src="/Week2/tex/0db515af88cbc2004fbdf00c3b278aec.svg?invert_in_darkmode&sanitize=true" align=middle width=60.17046089999999pt height=24.65753399999998pt/> and <img src="/Week2/tex/3c82528e2550142d1bdbaebdd77441a8.svg?invert_in_darkmode&sanitize=true" align=middle width=57.61642094999999pt height=24.65753399999998pt/>.

You can also use as range the standard deviation of <img src="/Week2/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/>.







.

# Standford Machine Learning - Week 02

#### Tags
* Author : AH Uyekita
* Title  :  _Linear Regression with Multiple Variables_
* Date   : 24/02/2019
* Course : Machine Learning
    * **Instructor:** Andrew Ng

***

## Multivariate Linear Regression

The Multivariate Linear Regression is a generalization of the Univariate Linear Regression. The Equation (1) shows it.

$$h_{\theta}(x) = \theta_0 \cdot \underbrace{1}_{x_0} + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \dots + \theta_n x_n \tag{1}$$

Where:

* $x_i$: It is the observations value, and;
* $\theta_j$: The feature/variable parameter;
* $n$: Number of features, and;
* $m$: Number of observations.

For convenience of notation, define $x_0$ equal to 1. Converting equation (1) in matrix notation.

$$h_{\theta}(x) = \begin{bmatrix} \theta_0 & \theta_1 & \dots & \theta_n \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_n \end{bmatrix} = \theta^{T}x \tag{2}$$

Where:

* $\theta^T$: Row vector of parameters, and;
* $x$: Column vector values.

### Gradient Descent in Multivariate Linear Regression

The process is almost the same, which differ is the quantity of the features. The Equation (3) shows the generic Cost Function.

$$J(\theta) = \frac{1}{2m} \sum_{i = 1}^{m} ( h_{\theta}(x^{(i)}) - y^{(i)} )^2 \tag{3}$$

Updating the parameters.

$$\theta_n := \theta_n - \alpha \frac{1}{m} \sum_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}  ) \cdot x_n^{(i)} \tag{4}$$

#### Feature Scaling

In some cases it is necessary to scale the features because the difference between ranges (of the features) could leverage the Gradient Descent convergence. It means if one feature has range from -3 to +3 and the other from -10.000 to +10.000, the difference is too high and the convergence process will last too much.

A way to understand it, is plotting a contours plot using the parameters as axis x and y, and the third axis as Cost Function. Figure 1 shows an example.

![Figure 1 - Contours Plot.](01-img/ml_week_02_01.png)

<center><em>Figure 1 - Contours Plot of Cost Function.</em></center><br>

Due to the skewness of this plot, the Gradient Descent will take several steps (iterations) doing a zig-zag by reach the center of the figure (where we supose it is the minimum value of $J(\theta)$).

On the other hand, If you performs the feature scaling, Figure 1 will change to seems like Figure 2.

![Figure 2 - Contours Plot Scaled.](01-img/ml_week_02_02.png)

<center><em>Figure 2 - Contours Plot Scaled.</em></center><br>

As a result of the feature scaling, the convergence process in much faster.

The simplest way to do it is substracting the average and dividing it by the range, as shown in equation (5).

$$x_{i,scale} = \frac{x_i - average(x_i)}{range} \tag{5}$$

Where:

* $x_i$: Feature $i^{th}$;
* $average(x_i)$: Average of all values of $x_i$, and;
* $range$: The subtraction of $max(x_i)$ and $min(x_i)$.

You can also use as range the standard deviation of $x_i$.

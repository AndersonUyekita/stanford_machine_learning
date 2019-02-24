# Standford Machine Learning - Week 01

#### Tags
* Author : AH Uyekita
* Title  :  _Introduction_
* Date   : 22/02/2019
* Course : Machine Learning
    * **Instructor:** Andrew Ng

***

## Machine Learning

Spam filter, Google Search Engine, Photo Tagging are all examples of Machine Learning algorithms. The Definition of Machine Learning coined by Arthur Samuel.

>"Field of study that gives computers the ability to learn without being explicitly programmed" -- <cite>Arthur Samuel</cite>

Tom Mitchell has an improved definition of Machine Learning.

>"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E." -- <cite>Tom Mitchell</cite>

Machine Learning could be divided into two big families:

* Supervised learning, and;
* Unsupervised learning.

Let's talk a bit more about these two groups.

#### Supervised learning

Here, we teach the algorithm how to do something based on past experience (database), which has the "right answers".

Figure 1 shows an example.

![Figure 1 - Supervised Learning](01-img/ml_week_01_01.png)

<center><em>Figure 1 - Supervised Learning Example task.</em></center><br>

The crosses are wrong answers and the circles the right.

>In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.
>
>Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

#### Unsupervised learning

In this case the Machine Learning algorithm do not have any ideia of the right answer.

>Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

An example of Unsupervised Learning is the Clusterization process, which aims to find patterns/structures in the data.

### Model Representation

Figure 2 shows an example of Model Representation.

![Figure 2 - General Model.](01-img/ml_week_01_02.png)

<center><em>Figure 2 - General Model.</em></center><br>

Where:

* Training Set: Dataset;
* Learning Algorithm: ML Algorithm;
* $h$: the output of the ML Algorithm.

The $h$, also called hypothesis, will take the `Size of house`as input to `estimate the price`.

### Linear Regression

The first classifier presented in this course is the Linear Regression because it is the easiest one. The equation (1) shows the formulation.

$$\text{Hypothesis: } h_{\theta} = \theta_0 + \theta_1 \cdot x \tag{1}$$

The optimization problem behind the Linear Regression is the minimization of the errors, which is the distance between the line estimated by the Linear Regression and the actual value. Recall, this is also called Cost Function.

#### Cost Function

Defining it in a equation (2).

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i = 1}^{m} (h_{\theta}^{(i)} - y^{(i)} )^2 \tag{2}$$

Where:

* $m$: The total number of observations (training dataset size);
* $\theta_0$: The intercept;
* $\theta_1$: The slope;
* $h_{\theta}^{(i)}$: The estimated value of the $i^{th}$ observation, and. It is also known as $\hat y^{(i)}$;
* $y^{(i)}$: The actual value of the $i^{th}$ observation.

So, my objective is to minimize the $J(\theta_0, \theta_1)$.

$$\text{Minimize: } J(\theta_0,\theta_1) \tag{3}$$

#### Gradient Descent

This is a strategy to solve many problems. Figure 3 shows an example of the behaviour of this algorithm.

![Figure 3 - Gradient Descent.](01-img/ml_week_01_03.png)

<center><em>Figure 3 - Gradient Descent.</em></center><br>

As you can see, the Gradient Descent is sensitive to the initial point, which means the results could be different if you choose different start points. All this is made, using the derivate of the $J$ to sinalize the direction for each step (iteration).

$$\theta_i := \theta_i - \alpha \frac{\partial}{\partial \theta_i} J(\theta_i) \tag{4}$$

Where:

* $\alpha$: Learning Reate;
* $\frac{\partial J(\theta_i)}{\partial \theta_i}$: Partial derivate.

The interpretation of the derivate is presented in Figure 4.

![Figure 4 - Partial Derivates.](01-img/ml_week_01_04.png)

<center><em>Figure 4 - Partial Derivates.</em></center><br>

The derivate aims to update the value of each parameter. Graphicaly, when teh derivate is positive, we need to decrease the $\theta$ substracting it from $\theta$. In case of negatives values of derivates, we need to increase the value of $\theta$.

**Learning Rate**

In a way to converge the algorithm, the Learning Rate aims to provide little steps toward to the local minimum. This is necessary because if the step is too big there is no convergence. Figure 5 shows it.

![Figure 5 - Learning Rate.](01-img/ml_week_01_05.png)

<center><em>Figure 5 - Learning Rate.</em></center><br>

Have in mind, according to the proximity to the local minimum the derivate value became smaller, which helps the convergence.

#### Gradient Descent in Linear Regression

Given the equations of Linear, Regression, Gradient Descent and the Cost Function of a Linear Regression.

* Linear Regression.

$$h_{\theta}=\theta_0 + \theta_1 x \tag{1}$$

* Cost Function.

$$J(\theta_0,\theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2 \tag{2}$$

* Gradient Descent for $i^{th}$ parameter.

$$\theta_i := \theta_i - \frac{\partial}{\partial \theta_i} J(\theta_i) \tag{4}$$

Let's decompose equation (4) to take a look in the derivate component.

* Partial for $j^{th}$.

$$\frac{\partial}{\partial \theta_j}J(\theta_0, \theta_1) = \frac{\partial}{\partial \theta_j} \begin{pmatrix} \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2 \end{pmatrix} = $$

$$= \frac{\partial}{\partial \theta_j} \frac{1}{2m} \sum_{i=1}^{m} ( \underbrace{\theta_0^{(i)} + \theta_1 x^{(i)}}_{h_{\theta}(x^{(i)}) } - y^{(i)} )^2 \tag{5}$$

Now, let's substitute the $j$ by 0 and 1 in equation (5).

* $j = 0$

$$\frac{\partial }{\partial \theta_1} J(\theta_0, \theta_1) = \frac{1}{m} \sum_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \tag{6}$$

* $j = 1$

$$\frac{\partial }{\partial \theta_0} J(\theta_0, \theta_1) = \frac{1}{m} \sum_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x^{(i)} \tag{7}$$

From equations (6) and (7), it is possible to update the equation (4).

$$\theta_0 := \theta_0 - \frac{1}{m} \sum_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \tag{8}$$


$$\theta_1 := \theta_1 - \frac{1}{m} \sum_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x^{(i)} \tag{9}$$

Finally, given equation (8) and (9), you can calculate all the Gradient Descent to the Linear Regression.

Bear in mind, the Linear Regression Cost Function is convex, so there is no local optima and the algorithm should find the global maximum (or a value very close to it).

**Batch Linear Regression**

This is the way to calculate the Linear Regression presented so far, all the point is used to fit the line, but there are other approachs, which will be discussed later in the course.

(**)


$$\begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n \end{bmatrix} :=
\begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n \end{bmatrix} -
\alpha \sum_{i=1}^m \begin{Bmatrix} \begin{pmatrix}
\begin{bmatrix} \theta_0 \ \theta_1 \dots \theta_n \end{bmatrix}
\begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_n \end{bmatrix} -
\begin{bmatrix} y_0 \\ y_1 \\ \vdots \\ y_n \end{bmatrix}\end{pmatrix}
\begin{bmatrix} x_0^{(i)} \\ x_1^{(i)} \\ \vdots \\ x_2^{(i)} \end{bmatrix}\end{Bmatrix} \tag{17}$$

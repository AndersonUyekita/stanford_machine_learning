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
* <img src="/Week1/tex/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode&sanitize=true" align=middle width=9.47111549999999pt height=22.831056599999986pt/>: the output of the ML Algorithm.

The <img src="/Week1/tex/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode&sanitize=true" align=middle width=9.47111549999999pt height=22.831056599999986pt/>, also called hypothesis, will take the `Size of house`as input to `estimate the price`.

### Linear Regression

The first classifier presented in this course is the Linear Regression because it is the easiest one. The equation (1) shows the formulation.

<p align="center"><img src="/Week1/tex/7985e39b3229acd0ff978302d903cda8.svg?invert_in_darkmode&sanitize=true" align=middle width=201.41720115pt height=14.611878599999999pt/></p>

The optimization problem behind the Linear Regression is the minimization of the errors, which is the distance between the line estimated by the Linear Regression and the actual value. Recall, this is also called Cost Function.

#### Cost Function

Defining it in a equation (2).

<p align="center"><img src="/Week1/tex/a44da21b9783a9dea06aba3908baead8.svg?invert_in_darkmode&sanitize=true" align=middle width=225.01263015pt height=44.89738935pt/></p>

Where:

* <img src="/Week1/tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/>: The total number of observations (training dataset size);
* <img src="/Week1/tex/1a3151e36f9f52b61f5bf76c08bdae2b.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/>: The intercept;
* <img src="/Week1/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/>: The slope;
* <img src="/Week1/tex/be05db7f7defcf4a5aee5ab374cdd019.svg?invert_in_darkmode&sanitize=true" align=middle width=24.39604694999999pt height=34.337843099999986pt/>: The estimated value of the <img src="/Week1/tex/3def24cf259215eefdd43e76525fb473.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32504519999999pt height=27.91243950000002pt/> observation, and. It is also known as <img src="/Week1/tex/549f15d8040f8d91862486e5cc44d716.svg?invert_in_darkmode&sanitize=true" align=middle width=23.57413739999999pt height=29.190975000000005pt/>;
* <img src="/Week1/tex/708d9d53037c10f462707daa2370b7df.svg?invert_in_darkmode&sanitize=true" align=middle width=23.57413739999999pt height=29.190975000000005pt/>: The actual value of the <img src="/Week1/tex/3def24cf259215eefdd43e76525fb473.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32504519999999pt height=27.91243950000002pt/> observation.

So, my objective is to minimize the <img src="/Week1/tex/dde9bb45048690d27e2b6c199faf1f5d.svg?invert_in_darkmode&sanitize=true" align=middle width=60.970370999999986pt height=24.65753399999998pt/>.

<p align="center"><img src="/Week1/tex/5805ce10e6eed1a2f1c1ddaf0b77647f.svg?invert_in_darkmode&sanitize=true" align=middle width=139.0527138pt height=16.438356pt/></p>

#### Gradient Descent

This is a strategy to solve many problems. Figure 3 shows an example of the behaviour of this algorithm.

![Figure 3 - Gradient Descent.](01-img/ml_week_01_03.png)

<center><em>Figure 3 - Gradient Descent.</em></center><br>

As you can see, the Gradient Descent is sensitive to the initial point, which means the results could be different if you choose different start points. All this is made, using the derivate of the <img src="/Week1/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> to sinalize the direction for each step (iteration).

<p align="center"><img src="/Week1/tex/8b4ab96a0759043c1bcfe8cb0347c6a2.svg?invert_in_darkmode&sanitize=true" align=middle width=146.97768194999998pt height=36.2778141pt/></p>

Where:

* <img src="/Week1/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>: Learning Reate;
* <img src="/Week1/tex/dc092b218ab95b0290c3c0a465726e66.svg?invert_in_darkmode&sanitize=true" align=middle width=37.82906325pt height=33.20539859999999pt/>: Partial derivate.

The interpretation of the derivate is presented in Figure 4.

![Figure 4 - Partial Derivates.](01-img/ml_week_01_04.png)

<center><em>Figure 4 - Partial Derivates.</em></center><br>

The derivate aims to update the value of each parameter. Graphicaly, when teh derivate is positive, we need to decrease the <img src="/Week1/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/> substracting it from <img src="/Week1/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>. In case of negatives values of derivates, we need to increase the value of <img src="/Week1/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>.

**Learning Rate**

In a way to converge the algorithm, the Learning Rate aims to provide little steps toward to the local minimum. This is necessary because if the step is too big there is no convergence. Figure 5 shows it.

![Figure 5 - Learning Rate.](01-img/ml_week_01_05.png)

<center><em>Figure 5 - Learning Rate.</em></center><br>

Have in mind, according to the proximity to the local minimum the derivate value became smaller, which helps the convergence.

#### Gradient Descent in Linear Regression

Given the equations of Linear, Regression, Gradient Descent and the Cost Function of a Linear Regression.

* Linear Regression.

<p align="center"><img src="/Week1/tex/199f3848fcfa8eb5d5480d4a511405e0.svg?invert_in_darkmode&sanitize=true" align=middle width=98.4947964pt height=13.881256950000001pt/></p>

* Cost Function.

<p align="center"><img src="/Week1/tex/7a4ee1040557ba521553fce9a415b5f3.svg?invert_in_darkmode&sanitize=true" align=middle width=254.63021925pt height=44.89738935pt/></p>

* Gradient Descent for <img src="/Week1/tex/3def24cf259215eefdd43e76525fb473.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32504519999999pt height=27.91243950000002pt/> parameter.

<p align="center"><img src="/Week1/tex/3efb2cadc5eb73c58c15950f955a16be.svg?invert_in_darkmode&sanitize=true" align=middle width=136.4011836pt height=36.2778141pt/></p>

Let's decompose equation (4) to take a look in the derivate component.

* Partial for <img src="/Week1/tex/95291b39ba5d9dba052b40bf07b12cd2.svg?invert_in_darkmode&sanitize=true" align=middle width=20.37223649999999pt height=27.91243950000002pt/>.

<p align="center"><img src="/Week1/tex/339c68bc6c4d197cf58ff429f95bbd4a.svg?invert_in_darkmode&sanitize=true" align=middle width=356.38612515pt height=38.5152603pt/></p>

<p align="center"><img src="/Week1/tex/8a928e94816217c94c815f3fcade143e.svg?invert_in_darkmode&sanitize=true" align=middle width=242.16569025pt height=56.79553274999999pt/></p>

Now, let's substitute the <img src="/Week1/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/> by 0 and 1 in equation (5).

* <img src="/Week1/tex/1d4e4268b1d33a4b6501ee8570a36775.svg?invert_in_darkmode&sanitize=true" align=middle width=37.84725779999999pt height=21.68300969999999pt/>

<p align="center"><img src="/Week1/tex/18da05f9c46f17ab081e76489045be55.svg?invert_in_darkmode&sanitize=true" align=middle width=266.56282124999996pt height=44.89738935pt/></p>

* <img src="/Week1/tex/99c0c04664ba70db8077b2b6415b2d85.svg?invert_in_darkmode&sanitize=true" align=middle width=37.84725779999999pt height=21.68300969999999pt/>

<p align="center"><img src="/Week1/tex/32cf718bec39bb4fe1cb87f75a0f509d.svg?invert_in_darkmode&sanitize=true" align=middle width=302.75472314999996pt height=44.89738935pt/></p>

From equations (6) and (7), it is possible to update the equation (4).

<p align="center"><img src="/Week1/tex/c8531bd937c34444bb9c4be8fc27eacb.svg?invert_in_darkmode&sanitize=true" align=middle width=233.72821725pt height=44.89738935pt/></p>


<p align="center"><img src="/Week1/tex/1da5f48936d637fa027628c218a870a0.svg?invert_in_darkmode&sanitize=true" align=middle width=269.92011915pt height=44.89738935pt/></p>

Finally, given equation (8) and (9), you can calculate all the Gradient Descent to the Linear Regression.

Bear in mind, the Linear Regression Cost Function is convex, so there is no local optima and the algorithm should find the global maximum (or a value very close to it).

**Batch Linear Regression**

This is the way to calculate the Linear Regression presented so far, all the point is used to fit the line, but there are other approachs, which will be discussed later in the course.

(**)


<p align="center"><img src="/Week1/tex/c231505cd701032cc1c7361422d357f0.svg?invert_in_darkmode&sanitize=true" align=middle width=464.03075069999994pt height=99.2791074pt/></p>

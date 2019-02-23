# Standford Machine Learning - Week 01

#### Tags
* Author : AH Uyekita
* Title  :  _Introduction_
* Date   : 22/02/2019
* Course : Machine Learning
    * **Instructor:** Andrew Ng

***

## Machine Learning

Machine Learning could be divided into two big families:

* Supervised learning, and;
* Unsupervised learning.

#### Supervised learning

Temos uma idéia de como será a saída (os resultados do algoritmo) mesmo que seja uma vaga idéia. a partir de uma série de dados reais alimentamos um algoritmo que pode ser uma aproximação linear, polinomial, etc. que ele produza mais valores ("Right answers" given dito no video). Neste bojo também se encontra os algoritmos de classificação, onde eles são responsáveis em classificar em valores discretos uma determinada entrada (baseado na idade e no tamanho do tumor, é câncer?)

#### Unsupervised learning

Diz se que é não supervisionado porque não se sabe de antemão qual é o algoritmo a ser usado (note que no supervisionado temos uma idéia de como será a saída e o algoritmo que será usado regressão/classificação).

$$\begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n \end{bmatrix} =
\begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n \end{bmatrix} -
\alpha \sum_{i=1}^m \begin{Bmatrix} \begin{pmatrix}
\begin{bmatrix} \theta_0 \ \theta\_1 \dots \theta\_n \end{bmatrix}
\begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_n \end{bmatrix} -
\begin{bmatrix} y_0 \\ y_1 \\ \vdots \\ y_n \end{bmatrix}\end{pmatrix}
\begin{bmatrix} x_0^{(i)} \\ x_1^{(i)} \\ \vdots \\ x_2^{(i)} \end{bmatrix}\end{Bmatrix} \tag{17}$$

a
a
a
a
a

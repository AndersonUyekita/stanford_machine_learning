Machine Learning `week01`
=========================

#### Tags

-   Author : AH Uyekita
-   Title: *Introduction*
-   Date : 15/01/2019
-   Course : Machine Learning
    -   **Instructor:** Andrew Ng

------------------------------------------------------------------------

Machine Learning
----------------

Machine Learning pode ser divido em:

-   Supervised learning, e;
-   Unsupervised learning.

**Supervised learning:** Temos uma idéia de como será a saída (os resultados do algoritmo) mesmo que seja uma vaga idéia. a partir de uma série de dados reais alimentamos um algoritmo que pode ser uma aproximação linear, polinomial, etc. que ele produza mais valores ("Right answers" given dito no video). Neste bojo também se encontra os algoritmos de classificação, onde eles são responsáveis em classificar em valores discretos uma determinada entrada (baseado na idade e no tamanho do tumor, é câncer?)

**Unsupervised learning:** Diz se que é não supervisionado porque não se sabe de antemão qual é o algoritmo a ser usado (note que no supervisionado temos uma idéia de como será a saída e o algoritmo que será usado regressão/classificação).

$$\\begin{bmatrix} \\theta\_0 \\\\ \\theta\_1 \\\\ \\vdots \\\\ \\theta\_n \\end{bmatrix} =
\\begin{bmatrix} \\theta\_0 \\\\ \\theta\_1 \\\\ \\vdots \\\\ \\theta\_n \\end{bmatrix} - 
\\alpha \\sum\_{i=1}^m\\begin{Bmatrix}\\begin{pmatrix}
\\begin{bmatrix} \\theta\_0 \\ \\theta\_1 \\dots \\theta\_n \\end{bmatrix}
\\begin{bmatrix} x\_0 \\\\ x\_1 \\\\ \\vdots \\\\ x\_n \\end{bmatrix} - 
\\begin{bmatrix} y\_0 \\\\ y\_1 \\\\ \\vdots \\\\ y\_n \\end{bmatrix}\\end{pmatrix}
\\begin{bmatrix} x\_0^{(i)} \\\\ x\_1^{(i)} \\\\ \\vdots \\\\ x\_2^{(i)} \\end{bmatrix}\\end{Bmatrix} \\tag{17}$$

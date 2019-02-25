# Standford Machine Learning - Week 03

#### Tags
* Author : AH Uyekita
* Title  :  _Logistic Regression_
* Date   : 25/02/2019
* Course : Machine Learning
    * **Instructor:** Andrew Ng

***

## Logistic Regression

É um dos algoritmos mais populares e mais usados.

Exemplos:

* Uso na classificação de emails: Spam ou não spam
* Verificação de transações do comercio online: Fraudulento ou não?
* Tumor: Maligno ou Benigno

A idéia é **classificar** as variáveis conforme a nomenclatura:

<p align="center"><img src="/Week3/tex/3a198f0ee92fe0bded8260eba30d512a.svg?invert_in_darkmode&sanitize=true" align=middle width=52.4846454pt height=13.789957499999998pt/></p>

Onde:

* 0: É a classe negativa (exemplo: tumor benigno)
* 1: É a classe positiva (exemplo: tumor maligno)

Há uma arbitratiedade na atribuição do que é 0 ou 1, mas há uma intuição de que a classe negativa (no nosso caso de exemplo tem valor de zero) transmite o sentido de **não ter alguma coisa** que no caso é não ter o tumor maligno. Ao passo que, a classe positiva transmite o sentido **de ter alguma coisa**, isto é, ter um tumor maligno.

Além da classificação binária há formas mais complexas, das quais envolvem multiclasses (que serão posteriormente discutida).

### Motivos para não usar Regressão Linear

![](01-img/1.png)

A classificação do problema acima é bem simples:

* Yes (1) : É maligno
* No  (0) : É benigno

Inicialmente, com uma abordagem simplista, pode-se usar a regressão como solução, conforme a figura 2.

![Figura 2](01-img/2.png)

Usa-se um limite em 0.5 separando pela metade o que é maligno e benigno. Aparentemente está tudo bem, mas ao estressar o modelo inserindo um "outlier" essa regressão pode ser alteradam conforme exibida na figura 3.

![Figura 3](01-img/3.png)

A idéia de traçar uma regressão linear simples não é a solução adequada visto que essa regressão será sensível ao outlier ou um ponto qualquer no extremo. Note que a linha azul que é a nova regressão após a inserção de novos pontos altera-se. Além disso, a adição de novas observações alterou a classificação de outras, onde se nota nitidamente que há dois grupos.

>Por este motivo não se deve usar a Regressão Linear para resolver problemas de classificação.

Lembre-se que para os problemas de regressão os valores de <img src="/Week3/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/> podem assumir valores que são maiores de um ou menores de zero.

Já num problema de classificação os valores de y só assumem dois valores discretos: 0 ou 1. Desta maneira, necessitamos de uma hipótese (<img src="/Week3/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/>) que fique compreendida entre esses dois limites (0 ou 1), onde qualquer valor acima de 0,5 seria 1 e todos abaixo desse valor 0.

### Logistic Regression

Nosso objetivo é ter uma hipótese entre 0 e 1.

<p align="center"><img src="/Week3/tex/655696d22d951956485f778629b478c5.svg?invert_in_darkmode&sanitize=true" align=middle width=77.18196255pt height=13.881256950000001pt/></p>

Dessa maneira, devemos "invertar" uma hiótese que sintetize isso.

Lembre-se que a hipótese da regressão linear é conforme a equação 1

<p align="center"><img src="/Week3/tex/e367a0f3beae52d618cfa63fe46c0af1.svg?invert_in_darkmode&sanitize=true" align=middle width=88.93044104999998pt height=18.7598829pt/></p>
Já para a Regressão Logistica usaremos uma função sobre <img src="/Week3/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/>.

<p align="center"><img src="/Week3/tex/51c3803c3c11b32b2958407ca56a75a5.svg?invert_in_darkmode&sanitize=true" align=middle width=110.1462318pt height=18.7598829pt/></p>
Onde:

* <img src="/Week3/tex/c9b8827f393c73fd6ffa72e5ee18bb61.svg?invert_in_darkmode&sanitize=true" align=middle width=92.04520709999998pt height=27.77565449999998pt/>

![Figura 4](01-img/4.png)

A função _Sigmoid_ ou Logística, possui duas retas assintóticas (0 e 1), das quais representam os nossos valores de y.

#### Interpretação

![Figura 5](01-img/5.png)

Deve-se interpretar os resultados da seguinte maneira:

<img src="/Week3/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/> = probabilidade estimada dado que `y = 1` com entradas x;

Exemplo:

Se <img src="/Week3/tex/232d50967b150e6873e8a97d94bf2bfe.svg?invert_in_darkmode&sanitize=true" align=middle width=239.9088648pt height=47.6716218pt/>.
Se <img src="/Week3/tex/f4cf86064b3a2f2fc938ed4ede301e65.svg?invert_in_darkmode&sanitize=true" align=middle width=90.23018564999998pt height=24.65753399999998pt/>, a probabilidade que o tumor seja maligno é de 70%.

A probabilidade de que y seja (y = 1), dado X e parametrizado por <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>. Que é equacionado como: <img src="/Week3/tex/52dc2d6fffdb13395a571f12ccdb7967.svg?invert_in_darkmode&sanitize=true" align=middle width=154.85520764999998pt height=24.65753399999998pt/>

<p align="center"><img src="/Week3/tex/4d7d421a5bed92eddd8c0681fca3f548.svg?invert_in_darkmode&sanitize=true" align=middle width=179.04453435pt height=16.438356pt/></p>

## Decision Boundary (Fronteiras de Decisão)

![Figura 6](01-img/6.png)

A figura 6 (acima) exemplifica o entendimento que devemos ter da função logística.

Para <img src="/Week3/tex/28bfa1de0b829a8ef9aebb8eb6eb92a3.svg?invert_in_darkmode&sanitize=true" align=middle width=38.78604674999999pt height=21.18721440000001pt/>: Teremos um valor de <img src="/Week3/tex/9008eec3f9c4d5100375da631c794630.svg?invert_in_darkmode&sanitize=true" align=middle width=29.58340934999999pt height=24.65753399999998pt/> maior que 0.5 (<img src="/Week3/tex/21c16c5ef58987af7eb64e6169fe5c16.svg?invert_in_darkmode&sanitize=true" align=middle width=72.5056827pt height=24.65753399999998pt/> ) quando z for maior ou igual a zero (<img src="/Week3/tex/c76afb5ecc4560fcb097577da327d2ad.svg?invert_in_darkmode&sanitize=true" align=middle width=38.50445939999999pt height=21.18721440000001pt/>). Lembre-se que o <img src="/Week3/tex/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode&sanitize=true" align=middle width=8.367621899999993pt height=14.15524440000002pt/> é o nosso <img src="/Week3/tex/45e726186266b4456810a6a4829609f8.svg?invert_in_darkmode&sanitize=true" align=middle width=27.92410829999999pt height=27.6567522pt/>, ou seja, sempre que <img src="/Week3/tex/45e726186266b4456810a6a4829609f8.svg?invert_in_darkmode&sanitize=true" align=middle width=27.92410829999999pt height=27.6567522pt/> for maior ou igual a zero o valor de y será 1.

<p align="center"><img src="/Week3/tex/88e990aaa755f34d5c8b13d8cf676f08.svg?invert_in_darkmode&sanitize=true" align=middle width=153.93257219999998pt height=16.438356pt/></p>

Substituindo <img src="/Week3/tex/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode&sanitize=true" align=middle width=8.367621899999993pt height=14.15524440000002pt/> por <img src="/Week3/tex/45e726186266b4456810a6a4829609f8.svg?invert_in_darkmode&sanitize=true" align=middle width=27.92410829999999pt height=27.6567522pt/>, tem-se:

<p align="center"><img src="/Week3/tex/ca791cd8913820b7e46b42796d5cf899.svg?invert_in_darkmode&sanitize=true" align=middle width=297.56790194999996pt height=18.7598829pt/></p>



Para <img src="/Week3/tex/54e8ccb1d0f9464f43edf5b1665c9763.svg?invert_in_darkmode&sanitize=true" align=middle width=38.78604674999999pt height=21.18721440000001pt/>: Seria o "inverso".

### Exemplo - Linear Decision Boudary

![Figura 7](01-img/7.png)

Supõe-se que os valores de <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/> são <img src="/Week3/tex/7797a64394cd0ba87c8777c9491c6170.svg?invert_in_darkmode&sanitize=true" align=middle width=42.92250929999999pt height=67.39784699999998pt/>.

Com base na equação (4), substituímos os valores de <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>.

<p align="center"><img src="/Week3/tex/e2ae782b395ba263fa92ab0b8069310b.svg?invert_in_darkmode&sanitize=true" align=middle width=375.92190074999996pt height=40.5953856pt/></p>

Note que o resultado da equação (5) é uma reta, conforme a equação (6) e figura 8.

<p align="center"><img src="/Week3/tex/db563a95aa230aba4842072d558611d0.svg?invert_in_darkmode&sanitize=true" align=middle width=83.76692609999999pt height=13.059335849999998pt/></p>
![Figura 8](01-img/8.png)

Dessa maneira, divide-se o espaço em duas partes por uma reta que é chamada de **decision boundary** que é equacionada por <img src="/Week3/tex/76e72e0c82a221e3186eb4c06af71305.svg?invert_in_darkmode&sanitize=true" align=middle width=83.76692609999998pt height=21.18721440000001pt/>. Deve-se enteder o seguinte:

* Para que y seja 1 (<img src="/Week3/tex/28bfa1de0b829a8ef9aebb8eb6eb92a3.svg?invert_in_darkmode&sanitize=true" align=middle width=38.78604674999999pt height=21.18721440000001pt/>) a relação deve ser verdadeira para <img src="/Week3/tex/ba8b61a0d1f13fe926f9089c36f810b2.svg?invert_in_darkmode&sanitize=true" align=middle width=83.76692609999998pt height=21.18721440000001pt/>;
* Para que y seja 0 (<img src="/Week3/tex/54e8ccb1d0f9464f43edf5b1665c9763.svg?invert_in_darkmode&sanitize=true" align=middle width=38.78604674999999pt height=21.18721440000001pt/>) a relação deve ser verdadeira para <img src="/Week3/tex/91e6d0ae5a7e6ce70df7d7756e7082db.svg?invert_in_darkmode&sanitize=true" align=middle width=83.76692609999998pt height=21.18721440000001pt/>;

### Exemplo - Non-linear Decision Boudary

Pode-se usar um _decision boundary_ que não seja linear, tal como um círculo que será explanada neste tópico e elucidado na Figura 9. Note que a hipótese (<img src="/Week3/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/>) possui elementos quadráticos.

![Figura 9](01-img/9.png)

Analogamente ao exemplo anterior, vamos supor a escolha dos seguintes:

<p align="center"><img src="/Week3/tex/1503e8e006ebb22851c123c0235b2e04.svg?invert_in_darkmode&sanitize=true" align=middle width=73.01366655pt height=98.63111444999998pt/></p>


Desta maneira, a hipótese para que y seja 1, será **se**:

<p align="center"><img src="/Week3/tex/70a2bc1603a72c8a6c779e10d11d0479.svg?invert_in_darkmode&sanitize=true" align=middle width=208.62968775pt height=18.2666319pt/></p>

Portanto, a minha _decision boudary_ será similar à equação (7), que pode ser resumoda como uma equação de cículo!

<p align="center"><img src="/Week3/tex/18a98f86a5907c8c27c420b1d7eee90a.svg?invert_in_darkmode&sanitize=true" align=middle width=83.76692609999999pt height=18.2666319pt/></p>

* Fora do círculo: `y = 1`;
* Dentro do círculo: `y = 0`.

A figura 10 apresenta o desenho final com o _decision boudary_.

![Figura 10](01-img/10.png)

Observe que o _decision boundary_ é calculado a partir dos parâmetros estabelecidos e não do _training set_.

## Cost Function

Recapitulando com a ajuda da figura 11.

![Figura 11](01-img/11.png)

O _Cost Function_ base será aquele apresentado na regressão linear, conforme a equação (9).

<p align="center"><img src="/Week3/tex/99ddb793f7119c42c9c8db268361e6fa.svg?invert_in_darkmode&sanitize=true" align=middle width=232.00001384999996pt height=44.89738935pt/></p>

Note que o `1/2` foi posto dentro do somatório (o que não altera nada já que é uma constante). Substituindo o somatório por uma outra função, temos:

<p align="center"><img src="/Week3/tex/08f17dde9fcb5fafef8f321d808d94d0.svg?invert_in_darkmode&sanitize=true" align=middle width=209.7216363pt height=32.990165999999995pt/></p>
Onde o <img src="/Week3/tex/717cf84d5ce7ea61ec631b70067afe6a.svg?invert_in_darkmode&sanitize=true" align=middle width=132.29095439999998pt height=29.190975000000005pt/> é:

<p align="center"><img src="/Week3/tex/4ec35b03bb2e7a98bfb7b2e4cc048f7e.svg?invert_in_darkmode&sanitize=true" align=middle width=285.03370335pt height=32.990165999999995pt/></p>

Para os problemas de regressão linear, a função <img src="/Week3/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/> é uma função convexa, isto é, ela não possui mínimos locais, o que atrapalha na convergência de uma solução usando o método do gradiente descendente. Para o caso da função logística a função de hipótese (<img src="/Week3/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/>) é uma função bem complicada, conforme a equação (11), e infelizmente essa função não é convexa.

<p align="center"><img src="/Week3/tex/b9f361352b7a0b46b1c3998d2d91cab4.svg?invert_in_darkmode&sanitize=true" align=middle width=133.19870024999997pt height=34.9287444pt/></p>

A Figura 12 apresenta os exemplos de desenhos de um <img src="/Week3/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/> `não convexo` e `convexo`.

![Figura 12](01-img/12.png)

Desta maneira, substitui-se o _Cost function_ da regressão linear por:

<p align="center"><img src="/Week3/tex/55bc79cc510b26b0d787192e9d5ab745.svg?invert_in_darkmode&sanitize=true" align=middle width=328.25765114999996pt height=49.315569599999996pt/></p>

Vamos analisar essas duas equação separadamente.

A Figura 13 apresenta o gráfico de comportamento da nova _Cost Function_ para `y = 1`.

![Figura 13](01-img/13.png)

Se estamos analisando a equação para `y = 1`, quando <img src="/Week3/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/> é 1, tem-se um valor de `Cost` igual à zero, mas caso o valor de <img src="/Week3/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/> esteja aproximando-se de zero o valor de `Cost` tenderá ao infinito. Isto captura a intuição de que <img src="/Week3/tex/11e1c33223daf43fe4a7083783f1c14a.svg?invert_in_darkmode&sanitize=true" align=middle width=69.22554374999999pt height=24.65753399999998pt/> (para o caso de `y = 1`) terá um <img src="/Week3/tex/234abe52d661f3172c5d1c7ced29b10b.svg?invert_in_darkmode&sanitize=true" align=middle width=123.69459134999997pt height=24.65753399999998pt/>. Além disso, terá um alto valor de `Cost` quando se aproximar de zero.

A Figura 14 apresenta o gráfico de comportamento da nova _Cost Function_ para `y = 0`.

![Figura 14](01-img/14.png)

Bom de maneira análoga, se estamos analisando a equação para `y = 0`, quando <img src="/Week3/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/> é zero, tem-se um valor de `Cost` igual à zero, mas no caso de o valor de <img src="/Week3/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/> esteja aproximando-se de 1 o valor de `Cost` tenderá ao infinito.

### Simplified Cost Function and Gradient Descent

A partir das equações (10) e (12), pode-se resumir a equação (12) para uma única equação.

<p align="center"><img src="/Week3/tex/da6f0be8668dedce7c717f7fd5de53a9.svg?invert_in_darkmode&sanitize=true" align=middle width=483.7789143pt height=19.526994300000002pt/></p>

Agora substituindo a equação (13) na equação (10).

<p align="center"><img src="/Week3/tex/3c26ee00262aeca9529617acd5199e06.svg?invert_in_darkmode&sanitize=true" align=middle width=440.79099899999994pt height=44.89738935pt/></p>

>O motivo de escolher essa função <img src="/Week3/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/> é devido ao Método de Máxima Verossimilhança. Usa-se esse método para estimar os parâmetros.

Logo, a partir da nova equação de <img src="/Week3/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/> o nosso objetivo é minimizar esse valor variando-se o <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>. Note que o novo _Cost Function_ possui as boas características de ser convexo. Portanto, pode-se usar a mesma técnica utilizada para a regressão linear que é o _Gradient Descent_, conforme elucida a figura 15.

![Figura 15](01-img/15.png)

A equação (15) apresenta o cálculo dos <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>.

<p align="center"><img src="/Week3/tex/24aa849d45114e2f1f775871221305ab.svg?invert_in_darkmode&sanitize=true" align=middle width=141.75611504999998pt height=38.5152603pt/></p>

Substituindo a derivada parcial de <img src="/Week3/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/>, tem-se:


<p align="center"><img src="/Week3/tex/ab8b4dad60381fe5866ad05cae4763d0.svg?invert_in_darkmode&sanitize=true" align=middle width=263.26525829999997pt height=44.89738935pt/></p>

Agora representando a equação de maneira matricial, onde j é a quantidade de parâmetros da hipótese.

<p align="center"><img src="/Week3/tex/bc547334e6053e32370c68f1bb39db1e.svg?invert_in_darkmode&sanitize=true" align=middle width=512.6525530499999pt height=100.82870325pt/></p>

## Advanced Optimization

É um capítulo que aborda basicamente o uso de funções `built-in` do Octave.

Inicialmente deve-se criar uma função que tenha como código algumas funções. Esta função será utillizada por uma segunda que fornecerá os dados definidos na primeira função.

* Derivadas parciais
* <img src="/Week3/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/>

## Multiclass Classification - One vs All Method

Este caso é a aplicação do algoritmo de _Logistic Regression_ tradicional, mas com um novo formato. O objetivo é marcar, por exemplo, emails como sendo de trabalho, amigos, família e hobbies, logo, são 4 classes.

A Figura 16 exemplifica um problema de Regressão Logística com 3 classes.

![Figura 16](01-img/16.png)


O idéia é simples e consiste na aplicação de várias Regressões Logísticas escolhendo uma classificação (classe positiva) e pondo o restante na classe negativa. Desta maneira, caso tenhamos 3 classificações, faremos 3 regressões logísticas, pois é uma combinação <img src="/Week3/tex/fcacdcd817003fbf2bf356a5d7af0adc.svg?invert_in_darkmode&sanitize=true" align=middle width=25.67585624999999pt height=22.465723500000017pt/> = <img src="/Week3/tex/4ba4a9430b5c576cdaea9d8a37895f89.svg?invert_in_darkmode&sanitize=true" align=middle width=32.42022794999999pt height=47.6716218pt/>.

## Regularization

Este capítulo discorrerá tanto sobre _Linear Regression_ e _Logistic Regression_, pois este conceito aborda ambas, adiante cada um deles terá o seu próprio sub-tópico.

O que é um problema de _Overfitting_ e de _Underfitting_?

A Figura 17, apresenta um resumo de tais tipos de problemas para a Regressão Linear.

![Figura 17](01-img/17.png)

O _Underfitting_ é um problema de falta de parâmetros e o _Overfitting_ é uma quantidade exagerada de parâmetros, este último pode ter até mesmo um <img src="/Week3/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/> igual à zero, mas possui muitas curvas em seu _decision boundary_ impossibilitando a generalização do modelo, sendo também não factível uma fronteira dessa maneira.

Note que estes mesmo conceitos de _Over_ ou _Underfitting_ podem ser levados para a Regressão Logística. A Figura 18 apresenta alguns exemplos para o caso de Regressão Logística.

![Figura 18](01-img/18.png)

Uma razão do _Overfitting_ que é latente: Quando a quantidade de observações (_training data_) é menor que das características (variáveis).

Há duas formas/opções de resolver esse problema:

* Reduzir a quantidade de variáveis;
       * Reduzir manualmente analisando caso a caso das variáveis e elegendo qual permanecerá e qual será descartado;
       * Usando o _selection algorithm_ que será apresentado adiante do curso;
* _Regularization_
       * mantém todas as características (variáveis), mas reduz a magnitude (os valores de $\theta_j$);
       * Funciona bem quando há muitas variáveis e cada um contribui um pouco para prever/estimar $y$.

Infelizmente, ao remover/eliminar uma variável haverá uma perda de informação para não incorrer nesse problema usa-se o _regularization_.

### _Cost Function_

Conforme já elucidado o _Overfitting_ é um problema quando há muitos parâmetros <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/> no modelo, o que gera um bom resultado no que tange o <img src="/Week3/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/>, pois o minimiza, contudo não é bom para generalizar devido à sua grande quantidade de curvas e também não é nada factível que a _decision boundary_ seja neste formato. A Figura 19 apresenta uma comparação entre um mesmo _data training_ e a hipótese adotada.

![Figura 19](01-img/19.png)

O problema de otimização ate então utilizado, isto é, a **função objetivo** é dado pela equação (18).

<p align="center"><img src="/Week3/tex/721a47ef27442a5c59205d4772e5c557.svg?invert_in_darkmode&sanitize=true" align=middle width=274.89280829999996pt height=44.89738935pt/></p>

Esta equação (18) que busca minimizar a soma dos quadrados será modificada ao inserir dois novos elementos (dois novos <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>s), mas com uma penalização para que eles <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>s sejam pequenos. A equação (19) elucida essa alteração.

<p align="center"><img src="/Week3/tex/719dbe67844c66c4e21cdfbe3ed7d8f2.svg?invert_in_darkmode&sanitize=true" align=middle width=519.64136235pt height=44.89738935pt/></p>

Observe que a função de minimização buscará minimizar os valores de <img src="/Week3/tex/ef3e4ae43ab69ed7bc41775203af5d03.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> e <img src="/Week3/tex/045f32cf246ff351495c9a128badf9e6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/>, pois o valor arbitrário `1000` sempre deixará <img src="/Week3/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/> elevado, para compensar isso os <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>s serão bem baixos. Isso mitigará os efeitos dos <img src="/Week3/tex/ef3e4ae43ab69ed7bc41775203af5d03.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> e <img src="/Week3/tex/045f32cf246ff351495c9a128badf9e6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/>, conforme elucida a Figura 20.

![Figura 20](01-img/20.png)

Os conceitos usados para _Regularization_ são:

* Pequenos valores para os parâmetros <img src="/Week3/tex/abc1e3127ccd6862dedd06f52b38fd66.svg?invert_in_darkmode&sanitize=true" align=middle width=112.25815424999998pt height=22.831056599999986pt/>;
       * Resultarão em hipóteses simples;
       * Menos propenso a ter um overfitting.

Desta maneira, para o exemplo de venda de casa que o _data training_ possuía 100 características (_features_ ou variáveis) e consequentemente outros 100 parâmetros. Há muitos <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>s e há um problema em qual desses muitos parâmetros/_features_ escolher para a regressão logística. Portanto, usa-se uma técnica que é aplicar o conceito supracitado de penalizar os parâmetros, mas em todos eles (do <img src="/Week3/tex/1a3151e36f9f52b61f5bf76c08bdae2b.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> ao <img src="/Week3/tex/6198455ff8721b0169e94091580d971b.svg?invert_in_darkmode&sanitize=true" align=middle width=15.842915549999992pt height=22.831056599999986pt/>). A nova equação <img src="/Week3/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/> é mostrada na equação (20).

<p align="center"><img src="/Week3/tex/da9d47720d2641619837db1601c2b2ea.svg?invert_in_darkmode&sanitize=true" align=middle width=481.3775769pt height=69.04177335pt/></p>

Observe que o <img src="/Week3/tex/1a3151e36f9f52b61f5bf76c08bdae2b.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> está fora do somatório (que está em **REGULARIZATION PARAMETER**) o que significa que ele não será afetado pela penalização. Isso é porque a influência dele estar ou não é tão pequena que não faz diferença nos resultados, além disso a **convenção** é não inserí-lo no somatório.

O <img src="/Week3/tex/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode&sanitize=true" align=middle width=9.58908224999999pt height=22.831056599999986pt/> é responsável pelo _trade-off_ entre dois objetivos:

* Treinar e obter bons valores de <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>, e;
       * Isso a partir da parcela da soma dos quadrados;
* Manter os parâmetros <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>s diminutos e evitar o _Overfitting_;
       * A partir do _Regularization Parameter_.

A Figura 21 apresenta um gráfico onde há duas curvas.

![Figura 21](01-img/21.png)

A linha em azul representa a regressão sem a regularização, ao passo que a linha rosa é aquela que possui a regularização. Note que a nova regressão é bem menos sinuosa, o que reflete mais a realidade e o torna generalista.

Note que poderá ocorrer casos de _underfitting_ se os valores de <img src="/Week3/tex/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode&sanitize=true" align=middle width=9.58908224999999pt height=22.831056599999986pt/> forem demasiadamente elevados, tornando os valores de <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>s tendendo à zero, sobrando então somente o <img src="/Week3/tex/1a3151e36f9f52b61f5bf76c08bdae2b.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/>, pois pela convenção ele não entra no _Regularization Parameter_. Uma outra maneira de se referir a esse tipo de problema é dizer que está com um forte viés ou "preconceito". A Figura 22 apresenta uma ilustração de como ficaria a regressão (uma reta paralela ao eixo abscissa).

![Figura 22](01-img/22.png)

Desta maneira, deve-se escolher o valor de <img src="/Week3/tex/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode&sanitize=true" align=middle width=9.58908224999999pt height=22.831056599999986pt/> cuidadosamente.

## _Regularized Linear Regression_

Especificamente sobre o _Regularized Linear Regression_ haverá uma pequena alteração no cálculo do <img src="/Week3/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/>, note que <img src="/Week3/tex/1a3151e36f9f52b61f5bf76c08bdae2b.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> será calculado de maneira distinta, pois ele não é afetado pelo _Regularization Parameter_ e os demais <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>'s serão simplificados. As equações (21) e (22) apresentam as novas formulações para este caso de regressão.

<p align="center"><img src="/Week3/tex/a5de4c4096b1f296df7bc8eead1c5099.svg?invert_in_darkmode&sanitize=true" align=middle width=274.01274615pt height=44.89738935pt/></p>

<p align="center"><img src="/Week3/tex/c9b992bb9b711160c281699d8a559692.svg?invert_in_darkmode&sanitize=true" align=middle width=756.74601915pt height=71.3447889pt/></p>
Portanto, para resolver esse problema pode-se recorrer ao _Gradient Descent_ como já foi utilizado na regressão linear "simples".

Para o caso de usar _Normal Equations_ há alguns benefícios de usar a _regularização_, pois ela lida bem com a não invertibilidade de <img src="/Week3/tex/bb00fc42f7162d614390dfafb5e7fcdd.svg?invert_in_darkmode&sanitize=true" align=middle width=40.17294764999998pt height=27.6567522pt/>, isto é, para o caso com regularização a matriz que será invertida será aquela com o parâmetro <img src="/Week3/tex/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode&sanitize=true" align=middle width=9.58908224999999pt height=22.831056599999986pt/> adicionado ao <img src="/Week3/tex/bb00fc42f7162d614390dfafb5e7fcdd.svg?invert_in_darkmode&sanitize=true" align=middle width=40.17294764999998pt height=27.6567522pt/>, a equação (23) ilustra isso.

<p align="center"><img src="/Week3/tex/5cb8978eba2e94b59d436333a067355b.svg?invert_in_darkmode&sanitize=true" align=middle width=385.75258095pt height=112.3841532pt/></p>

## _Regularized Logistic Regression_

Analogamente à _Regularized Linear Regression_ para o caso da Regressão Logística Regularizada será adicionado ao _Cost Function_ o parâmetro para penalizar os <img src="/Week3/tex/25f963e4038e147c49abe27e2bc7c560.svg?invert_in_darkmode&sanitize=true" align=middle width=37.68660224999999pt height=22.831056599999986pt/>s. A Figura 24 apresenta um _recap_.

![Figura 24](01-img/24.png)

Note que a nova equação para o cálculo do _Cost Function_ será de acordo com a equação (24).

<p align="center"><img src="/Week3/tex/64ab250bd0cddcffae67b952e000acd5.svg?invert_in_darkmode&sanitize=true" align=middle width=556.2911508pt height=72.9886014pt/></p>
Aplicando-se a formula supracitada (24), tem-se como resultado um gráfico (linha rosa) conforme ilustrado na Figura 25.

![Figura 25](01-img/25.png)

A atualização dos <img src="/Week3/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>s seguirá um roteiro parecido com o modelo regularizado de regressão linear, tendo somente que ressaltar que a hipótese (<img src="/Week3/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/>) é uma função sigmóide (ou logística), sendo assim uma equacionado parecido, mas não igual à regressão linear regularizada. A figura 26 apresenta as equações.

![Figura 26](01-img/26.png)

O uso de algoritmos de otimização avançados, pode ser feita a partir da seguinte rotina descrita na Figura 27.

![Figura 27](01-img/27.png)

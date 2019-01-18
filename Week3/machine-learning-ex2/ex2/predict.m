function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

% Cálculo de z
z = X * theta;

% Cálculo da hipótese h_theta(x)
p = 1./(1 + e.^(-z));

% Análise de p para atribuir 1 se for acima de 0.5 e
% atribuir 0 se for abaixo de 0.5

# Encontra os elementos que são menores que 0.5
h_0 = find(p < 0.5);

# Substitui esses valores menores que 0.5 por zero.
p(h_0) = 0;

# Encontra os elementos que são maiores que 0.5
h_1 = find(p >= 0.5);

# Substitui esses valores maiores que 0.5 por 1.
p(h_1) = 1;

% =========================================================================


end

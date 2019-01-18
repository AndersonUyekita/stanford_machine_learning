function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Exercício parecido com o CostFunction.m anterior.

% Calculo de z, só separei para a função ficar menor e não tão comprida
z = X * theta

% Uso do operador .* novamente para fazer o produto entre os elementos e
% não o produto matricial.
% Aplicação da fórmula da página 4 do ex2.pdf.
% Observação: Cálculo em duas etapas
% 1) é a mesma equação sem ser regularizada
% 2) é para deixar claro que para regularizar é necessário apenas agregar um parâmetro

% 1a Etapa
J = (1/m) * sum( -y .* log(sigmoid(z)) - (1 - y) .* log(1 - sigmoid(z)));

% 2a Etapa: NÃO ATUALIZAR O THETA_0
J = J + lambda/(2 * m) * theta(2:length(theta))' * theta(2:length(theta))

% Atualização do grad também será em duas etapas

% 1a etapa é igual para todos
grad = (1/m) * X' * (sigmoid(z) - y);

% 2a somente o theta_0 que não tem essa parcela: NÃO ATUALIZAR O THETA_0
grad = grad + [0;(lambda / m) * theta(2:length(theta))];


% =============================================================

end

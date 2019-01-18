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

% Exerc�cio parecido com o CostFunction.m anterior.

% Calculo de z, s� separei para a fun��o ficar menor e n�o t�o comprida
z = X * theta

% Uso do operador .* novamente para fazer o produto entre os elementos e
% n�o o produto matricial.
% Aplica��o da f�rmula da p�gina 4 do ex2.pdf.
% Observa��o: C�lculo em duas etapas
% 1) � a mesma equa��o sem ser regularizada
% 2) � para deixar claro que para regularizar � necess�rio apenas agregar um par�metro

% 1a Etapa
J = (1/m) * sum( -y .* log(sigmoid(z)) - (1 - y) .* log(1 - sigmoid(z)));

% 2a Etapa: N�O ATUALIZAR O THETA_0
J = J + lambda/(2 * m) * theta(2:length(theta))' * theta(2:length(theta))

% Atualiza��o do grad tamb�m ser� em duas etapas

% 1a etapa � igual para todos
grad = (1/m) * X' * (sigmoid(z) - y);

% 2a somente o theta_0 que n�o tem essa parcela: N�O ATUALIZAR O THETA_0
grad = grad + [0;(lambda / m) * theta(2:length(theta))];


% =============================================================

end

function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% Calculo de z, só separei para a função ficar menor e não tão comprida
z = X * theta

% Uso do operador .* novamente para fazer o produto entre os elementos e
% não o produto matricial.
% Aplicação da fórmula da página 4 do ex2.pdf.
J = (1/m) * sum( -y .* log(sigmoid(z)) - (1 - y) .* log(1 - sigmoid(z)));

% Atualização do grad
grad = (1/m) * X' * (sigmoid(z) - y);

% =============================================================

end

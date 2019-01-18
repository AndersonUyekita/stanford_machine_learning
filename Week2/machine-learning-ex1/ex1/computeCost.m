function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% X: É uma matrix que possui a primeira coluna repleta de 1
% y: São os valores observados


% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = ( (1)/(2*m) )*sum((X * theta - y ).^2)

% X é uma matrix m x 2
% theta é um vetor 2 x 1
% O produto X * theta é um vetor m x 1 que tem o mesmo comprimento
% do vetor y

% =========================================================================

end

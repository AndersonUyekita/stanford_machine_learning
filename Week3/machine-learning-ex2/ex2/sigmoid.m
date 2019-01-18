function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% Função simples. Só deve ter cuidado para o uso do operador ./ e .^
% estes dois operadores realizam as contas como se fossem por elemento e
% não matricialmente
g = 1./(1 + e.^(-z));

% =============================================================

end

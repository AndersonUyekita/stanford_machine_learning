function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Localiza os elementos que são iguais a 1, o resultado
% é um vetor com os endereços que y == 1 é TRUE
pos = find(y == 1);

% Localiza os elementos que são iguais a 0, o resultado
% é um vetor com os endereços que y == 0 é TRUE
neg = find(y == 0);

% Plotting somente os casos que y == 1
plot(X(pos, 1), X(pos, 2),'k+','LineWidth', 2,...
'MarkerSize', 7);

% Plotting somente os casos que y == 0
plot(X(neg, 1), X(neg, 2),'ko','MarkerFaceColor','y',...
'MarkerSize', 7);





% =========================================================================



hold off;

end

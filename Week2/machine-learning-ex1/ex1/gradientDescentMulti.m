function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_prov = zeros(1, size(X, 2));

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    % The theta must be in column so I dicided to transponse the sum
    % For further details see the lecture Gradient Descendet for Multiple Variables
    theta_prov = theta - alpha * (1/m) * sum((X * theta - y) .* X)'

    % Ensure one iteration -> Assignment the first calculation
    if iter == 1; % Special condition
        theta = theta_prov;      
    else iter > 1 % regular condition when number of iteration is above from 1
        if (computeCost(X, y, theta) > computeCost(X, y,theta_prov))
            theta = theta_prov; % Update old values
        else
            break % Stop condition
        end
    endif

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end

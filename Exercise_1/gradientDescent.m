function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % ==== cocmpute the gradient descent using matrices ====
    
    % The hypothesis is a vector, formed by multiplying the X matrix and 
    % theta vector. X has size (m x n), and theta is (n x 1), 
    % so the product is (m x 1) the same size as 'y'. 
    h = X * theta;
    
    % "errors vector" is the difference between the 'h' and 'y' vectors.
    error = h - y;
    
    % The change in theta (the "gradient") is the sum of the product of X 
    % and the "errors vector", scaled by alpha and 1/m. 
    % Since X is (m x n), and the error vector is (m x 1), 
    % and the result we want is the same size as theta (which is (n x 1), 
    % we need to transpose X before multiplying it by the error vector.
    % 
    % The vector multiplication automatically includes calculating 
    % the sum of the products so no need to use sum(X'*error)
    theta_change = (alpha / m) * (X' * error); % the gradient    
    
    % Subtract "change in theta" from the original value of theta. 
    theta = theta - theta_change;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    J_history(iter)

    
end

end

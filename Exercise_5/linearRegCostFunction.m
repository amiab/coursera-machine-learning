function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% use vectorised implementation avoiding loop   

% **** part 1 ****

% compute a vector 'h' containing all of the hypothesis values - one 
% for each training example (i.e. for each row of X). 
% X already includes the ones for the x0 
h = X * theta;

% now that the theta is used to calculate the hypothesis we can 
% set theta(1) to 0 to force it to zero so it doesn't add to the 
% regularization terms in regularised J & Grad.
theta(1) = 0;

% compute the error/difference between the hypothesis and y 
error = h - y;

% compute the square of each of those error terms 
error_sqr = error .^ 2;

% compute the sum of the error_sqr vector, and scale the result (multiply) 
% by 1/(2*m). That completed sum is the cost value J.

% when scaling by alpha and 1/m, use enough sets of parenthesis to get 
% the factors correct.

% unregularised cost function J 
cost = sum( error_sqr ) * 1/(2 * m);

% cost regularization term calculating the sum of the squares of theta 
% since theta(1) has been set to zero, it does not contribute to the 
% regularization term. scale the term by (lambda / (2 * m))
cost_reg_term = ( lambda / ( 2 * m ) ) * sum( theta .^ 2 );

% regularised cost function J 
J = cost + cost_reg_term;

% **** part 2 ****

% unregularised gradient 
grad_unreg = (( 1 / m ) * X' * error);

% The regularized gradient term is theta scaled by (lambda / m). 
% since theta(1) has been set to zero, it does not contribute to the 
% regularization term.
grad_reg_term = ( lambda / m ) * theta; 
% regularised gradient 
grad = grad_unreg + grad_reg_term;



% =========================================================================

grad = grad(:);

end

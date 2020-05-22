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

% we use regularisation to penalise large value of theta
% regularised cost calculation can be vectorised 

% step 1 same as in costFunction() we didtribute the summation operation 
% to find the cost
h = sigmoid( X * theta ); 
product_1 = y .* log( h );
product_2 = ( 1 - y ) .* log( 1 - h );
cost = ( 1 / m) * sum( - product_1 - product_2 );

% step 2 add lambda term explicity excluding bias term theta index 0
% Matlab start indexing from 1 (not zero)
% we can set theta(1) to zero. Since we already calculated h, and theta 
% is a local variable, we can modify theta(1) without causing any problems.
% or create a temp variable
theta_exclude_0_index = theta;
theta_exclude_0_index(1) = 0;

% or we can just select all elements from index 2 (theta 1)
% theta_exclude_0_index = theta(2:end); 
% if use this option we cannot use theta_exclude_0_index to calculate 
% grad_reg_term later because it will be short of 1 element. then to
% calculate the gradient see 2nd option at the end 


% cost regularization term calculating the sum of the squares of theta excl zero
% scaling the term by (lambda / (2 * m))
cost_reg_term = ( lambda / ( 2 * m ) ) * sum( theta_exclude_0_index .^ 2 );
% we could also use theta_exclude_0_index' * theta_exclude_0_index
% instead of sum( theta_exclude_0_index .^ 2 )
% multiply a vector by itself with transposition we will calculate the sum 
% automatically. if v = [1;4;8;9] ==> sum(v .^2) == v' * v
% note: use enough sets of parenthesis to get the correct result for example 
% 1/(2*m) and (1/2*m) give drastically different results and give hight cost value 

                
% step 3 add unregularized and regularized cost terms together to get J 
J = cost + cost_reg_term;


% step 4 Calculate the gradient with regularization while excluding theta 0.

error = h - y;

grad_reg_term = ( lambda / m ) * theta_exclude_0_index; 

grad = (( 1 / m ) * X' * error) + grad_reg_term;

% or calculate grad_reg_term with the original theta then assign a zero
% to grad_reg_term(1)=0 before adding the grad_reg_term to the grad


% =============================================================

end

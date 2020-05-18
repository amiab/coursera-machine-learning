function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% ==== cocmpute the cost function using matrices ==== 

% compute a vector 'h' containing all of the hypothesis values - one 
% for each training example (i.e. for each row of X). 
% The hypothesis (prediction) is simply the product of X and theta. 
h = X * theta;

% compute the difference between the hypothesis and y - 
% that's the error for each training example. Difference means subtract.
error = h - y;

% compute the square of each of those error terms 
error_sqr = error .^ 2;

% compute the sum of the error_sqr vector, and scale the result (multiply) 
% by 1/(2*m). That completed sum is the cost value J.
J = sum( error_sqr ) * 1/(2 * m);



% =========================================================================

end

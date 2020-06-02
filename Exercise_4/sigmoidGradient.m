function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

% the sigmoid gradient function can be computed as follow:
% g(z)' = d/dz g(z) = g(z)(1-g(z)) - the partial derivative of g(z) which 
% is 1/1-e^-z

sigmoid_z = sigmoid(z);
g = sigmoid_z .* ( 1 - sigmoid_z );  

% we don't want to automatically calculate the sum of the products, so 
% element-wise multiplication is needed here 











% =============================================================




end

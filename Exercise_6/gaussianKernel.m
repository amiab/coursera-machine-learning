function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

% use Gaussian kernel to find nonlinear decision boundaries with the SVM
% we can think of the Gaussian kernel as a similarity function that measures 
% the 'distance' between a pair of examples x_i & x_j
% The Gaussian kernel is also parameterized by a bandwidth parameter sigma
% which determines how fast the similarity metric decreases (to 0) as the 
% examples are further apart.

sum_sqr_distance = sum( (x1 - x2) .^ 2 );

sigma_term = 2 * sigma ^2;

exponent = sum_sqr_distance / sigma_term; 

sim = exp( - exponent);

% =============================================================
    
end

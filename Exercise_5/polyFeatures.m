function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

% method #1 use bsxfun() function, with @power operator
% binary singleton expansion function bsxfunc(FUNC, A, B)
% Apply element-wise binary operation specified by the function FUNC (@power) 
% to arrays A and B.

% X is a column vector of the feature and p is a row vector of exponents 
% from 1 to 'p'
X_poly = bsxfun(@power, X, [1:p]);

% method #2 use element-wise '.^' - Non vectorized
%for i = 1:p
    %X_poly(:,i) = X .^ i;
%end

% method #3 use element-wise '.^' - Vectorized implementation
%P = [1:p];
%X_poly = X .^ P;

% =========================================================================

end

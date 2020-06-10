function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%


% To perform anomaly detection, we 1st need to fit a model to the data's 
% distribution. Given the training set we want to estimate the Gaussian 
% distribution for each example features. 

% For each feature we need to find parameters mu and sigma square 

mu = mean(X);
% replicate mu to create a matrix
mu_matrix = repmat(mu, m, 1);
sigma2 = (1 / m) * sum((X - mu_matrix).^2);




% =============================================================


end

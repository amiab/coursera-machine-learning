function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
% mean() to get the mean for each column of X. 
% These are returned as row vectors (1 x n)
mu = mean(X); % returns a row vector

% sigma() to get the std deviation for each column of X. 
% These are returned as row vectors (1 x n)
sigma = std(X);

% apply mean & std values to each element in every row of the X matrix. 
% duplicate these vectors for each row in X so they're the same size.
% create a column vector of all-ones - size (m x 1) - and multiply it by 
% the mu or sigma row vector (1 x n). 
% Dimensionally, (m x 1) * (1 x n) gives a (m x n) matrix, and every row 
% of the resulting matrix will be identical.
m = size(X, 1); % returns the number of rows in X

mu_matrix = ones(m, 1) * mu;

sigma_matrix = ones(m, 1) * sigma;

% Now that X, mu, and sigma are all the same size, use element-wise 
% operators to compute X_normalized.
% subtract the mu matrix from X, and divide element-wise by the sigma 
% matrix, and arrive at X_normalized.

% keyboard; % for debugging

X_norm = (X - mu_matrix) ./ sigma_matrix; % Vectorized


% ============================================================

end

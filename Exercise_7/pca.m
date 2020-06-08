function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

% note: Matrix X features have already been normalized

% to compute the dataset principal components start by calculating 
% the covariance matrix of the data, which is given by:
sigma = (1/m) * (X' * X);

%  now run SVD on sigma to compute the principal components 
[U, S, V] = svd(sigma); 

% U contain the principal components (eigenvector) and S contain a diagonal matrix

% =========================================================================

end

function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% iterating through the centroids runs considerably faster than looping 
% through the training examples.

% Create a "distance" matrix of size (m x K) and initialize it to all zeros. 
% 'm' is the number of training examples, K is the number of centroids.
m = size(X, 1);
distance = zeros(m, K);

% Use a for-loop over the 1:K centroids.
for i = 1:K

    % keyboard;       
    
    % create a column vector of the distance 'd' from each training example 
    % to that centroid 
    
    % use bsxfun() to calculate the differences between each row in the 
    % X matrix and a centroid. Apply function @minus between X matrix & 
    % centroid row vector
    difference = bsxfun( @minus, X, centroids( i, : ) );
    
    % use sum() to calculate the sum of the squares of the differences in 
    % each row. use dim = 2 to operate on rows, sum of the rows not the columns
    d = sqrt( sum( difference .^ 2, 2) );
    
    % store distance as a column of the distance matrix.  
    % no need to calculate the square root - save on computation
    distance( :, i) = d;
        
end 

% after for-loop ends, we have a matrix of centroid distances.
% return indexes of the locations with the minimum distance as vector #
% (m x 1) with the indexes of the closest centroids.
% dim=2 returns a column vector containing the smallest element in each row.
[ ~, idx ] = min( distance, [], 2);



% =============================================================

end


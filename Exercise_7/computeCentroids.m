function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% "idx" is a vector with one entry for each example in "X", that tells us 
% which centroid each example is assigned to

% use for-loop over the range of centroid 1 - k.
for i = 1:K
    % use logical arrays to find the indexes of the examples row closest 
    % to this centroid i 
    selected_example = find(idx == i);
    
    % compute the mean of all these selected examples & assign it as the 
    % new centroid value
    centroids(i, :) = mean( X(selected_example, :));
    
    % centroids(i, :) = mean( X(x_row_selection, :),1 );
    % note: by default mean returns the mean of the elements in each column 
    % hpwever using dim=1 as in mean(...,1) causes the mean to be computed 
    % over the rows in the event that there is a centroid that has only one example.
end






% =============================================================


end


function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, 1:k);
%

% now we use eigenvectors returned by PCA, to reduce the feature dimension of 
% our dataset by projecting each example onto a lower dimensional space,  
% from 2D to 1D. 

% In practice, we use the projected data instead of the original data to 
% train our model faster as there are less dimensions in the input.
        
% project each example in X onto the top K components in U given by the first 
% K columns of U, that is 
U_reduce = U(:, 1:K);
Z = X * U_reduce;


% =============================================================

end

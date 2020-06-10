function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% compute the cost function and gradient for collaborative filtering. 
% parameters we are trying to learn are X and Theta. 
% In order to use an off-the-shelf minimizer such as fmincg, the cost function has 
% been set up to unroll the parameters into a single vector params

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% use vectorized implementation to compute J, since it will later be called 
% many times by the optimization package fmincg.
% using nested for loops slow down the run 

% predicted movie ratings for all users using the product of X and Theta. 
% A transposition is needed. dimensions of the result (movies x users).
h_prediction = X * Theta';  

% movie rating error by subtracting Y from the predicted ratings.
error = h_prediction - Y;

% "error_factor" by multiplying element-wise the movie rating error by the 
% R matrix. The error factor will be 0 for movies that a user has not rated. 
error_factor = error .* R;

% Calculate the unregularized cost as a scaled sum (sum(sum()) of the squares 
% of all of the terms in error_factor. The result should be a scalar.
J = (1/2) * sum( sum(error_factor .^ 2) );

% ---------------------- TEST & SUBMIT -------------------------

% Calculate the gradients based on the formulas in the ex8,pdf

% X gradient is the product of the error factor and the Theta matrix. 
% The sum is computed automatically by the vector multiplication
X_grad = error_factor * Theta;
 
% Theta gradient is the product of the error factor transposed and the X matrix. 
% The sum is computed automatically by the vector multiplication. 
Theta_grad = error_factor' * X;

% ---------------------- TEST & SUBMIT -------------------------

% Calculate the regularized cost:

% compute the regularization term as the scaled sum of the squares of all terms 
% in Theta and X. The result should be a scalar. 
% Note that for Recommender Systems there are no bias terms, so regularization 
% should include all columns of X and Theta.
J_reg_term_1 = (lambda / 2) * sum( sum( Theta .^ 2 ) ) ;
J_reg_term_2 = (lambda / 2) * sum( sum( X .^ 2 ) ) ;

% Add regularized and un-regularized cost terms.
J = J + J_reg_term_1 + J_reg_term_2;

% ---------------------- TEST & SUBMIT -------------------------

% Calculate the gradient regularization terms 

% X gradient regularization is the X matrix scaled by lambda.
X_grad_reg_term =  lambda * X;

% Theta gradient regularization is the Theta matrix scaled by lambda.
Theta_grad_reg_term = lambda * Theta;

% Add the regularization terms to their unregularized values.
X_grad = X_grad + X_grad_reg_term;
Theta_grad = Theta_grad + Theta_grad_reg_term;

% ---------------------- TEST & SUBMIT -------------------------


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end

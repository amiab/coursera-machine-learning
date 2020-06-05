function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% use cross validation data Xval, yval to determine the best C & sigma to use
% using suggested values below in multiplicative steps we will try all possible 
% pairs of values for C & sigma. if we try each of the 8 values listed blow 
% we will end up training and evaluating a total of 8^2 = 64 different models. 
% finally determine & return the best C & sigma parameters to use
            

C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% create a blank results matrix
results = zeros(length(C_list) * length(sigma_list), 3);
% row counter for the results matrix    
row = 1;

for C_val = C_list

    for sigma_val = sigma_list
        
        % Train the model using svmTrain with X, y, a value for C, and the gaussian 
        % kernel using a value for sigma. 
        model = svmTrain(X, y, C_val, @(x1, x2) gaussianKernel(x1, x2, sigma_val));
        
        % use svmPredict function to generate the predictions for the cross 
        % validation Xval with model 
        predictions = svmPredict(model, Xval);
        
        % Note: to evaluate the error on the cross validation set, recall for 
        % classification, the error is defined as the fraction of the cross 
        % validation examples that were classified incorrectly. 
        % In MATLAB, we can compute this error using mean(double(predictions ~= yval)), 
        % where predictions is a vector containing all the predictions from the SVM, 
        % and yval are the true labels from the cross validation set. 
        
        % Compute the error 'err_val' between your predictions and yval.
        error_val = mean(double( predictions ~= yval ));

        % for each error computation, save C, sigma, and error_val in the results matrix. 
        results(row,:) = [C_val sigma_val error_val];
        row = row + 1;
    end
end

% When all 64 computations are completed, use min() to find the row with 
% the minimum error, then use that row index to retrieve the C and sigma 
% values that produced the minimum error.

% use min() on the result matrix to find the row index and the lowest value 
% in the column #3 where error_val are stored 
% use the index to return C & sigma values that give the lowest validation error
[lowest_error_val idx] = min(results( :, 3 )); 

C = results( idx, 1 );
sigma = results( idx, 2 ); 

% keyboard;

% best C & sigma : ans is C = 1, sigma = 0.1



% =========================================================================

end

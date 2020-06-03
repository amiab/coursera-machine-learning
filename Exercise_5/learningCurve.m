function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

% generate learning curve useful in debugging learning algorithms. 
% learning curve plots training and cross validation error as a function of 
% training set size. 
% learningCurve.m returns a vector of errors for the training set and cross 
% validation set

% To plot the learning curve, we need a training and cross validation set error 
% for different training set sizes. 
% To obtain different training set sizes, we use different subsets of the 
% original training set X. 
% for a training set size of i, we use the first i examples (i.e., X(1:i,:) and y(1:i)). 
% we use the trainLinearReg function to find the parameters theta. 
% Note lambda is passed as a parameter to the learningCurve function. 
% After learning the parameters, we compute the error on the training and cross 
% validation sets. 

% the training error does not include the regularization term. 
% use linearRegCostFunction and set lambda to 0 only when using it to compute 
% the training error and cross validation error. 
% When computing the training set error, make sure you compute it on the 
% training subset (i.e., X(1:i,:) and y(1:i), instead of the entire training set). 
% However, for the cross validation error, compute it over the entire cross validation set. 
% store the computed errors in the vectors error_train and error_val. 

% note our dataset is divided into 3 parts 
% training set that our model will learn on: X, y
% cross validation set for determining the regularization parameter: Xval, yval
% test set for evaluating performance. These are 'unseen' examples which our 
% model did not see during training: Xtest, ytest
        
for i = 1:m
    % increase training set size by one for each iteration through the training set.
    Xtrain = X(1:i, :);
    ytrain = y(1:i); 
    
    % learn theta vector for the current size of training set
    theta = trainLinearReg(Xtrain, ytrain, lambda); 
    
    % pass lambda parameter as zero - we could do without ~ because the
    % 1st element that is returned is J
    % use subset
    [error_train(i), ~] = linearRegCostFunction(Xtrain, ytrain, theta, 0);
    % use entire set
    [error_val(i), ~] = linearRegCostFunction(Xval, yval, theta, 0);

end

% Tom Mosher's Tips:
% 
% 1/ Use the lambda parameter - from the learningCurve() parameter list - every 
% time you call trainLinearReg().

% 2/ Do not set lambda = 0 inside the learningCurve() function. You are going 
% to experiment with different non-zero lambda values in ex5.m, and the submit 
% grader doesn't use lambda = zero either. So do not hard-code lambda = 0 
% inside the learningCurve() function.
% 
% 3/ When you compute the training set error and the validation set error, 
% use your cost function with a zero for the lambda parameter. We want to 
% measure the error in the hypothesis, without including any additional 
% penalties for the theta values.
% 
% 4/ When you run the "ex5" script, you may get some "divide by zero" warnings. 
% These are expected and normal. fmincg() generates "divide by zero" warnings 
% whenever the training set has only one or two examples. Do not worry about it.



% -------------------------------------------------------------

% =========================================================================

end

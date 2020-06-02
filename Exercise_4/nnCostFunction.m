function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% outline forward propagation process. 
% duplicate some of the work in Ex3 because computing the gradients requires 
% some of the intermediate results from forward propagation. 
% y values in ex4 are a matrix, instead of a vector. This changes the method 
% for computing the cost J.

% add bias unit to input layer / add a column of 1's to X --> layer a1 
a1 = [ones(m,1) X]; % 5000x401
z2 = a1 * Theta1';  % (5000 x 401)*(401x25)
% actibvate z2
a2 = sigmoid(z2);   % 5000x25

% add bias unit to layer a2
a2 = [ones(m,1) a2];   % 5000x26  
z3 = a2 * Theta2';      % (5000x26) * (26x10)
% activate z3
a3 = sigmoid(z3);       % 5000x10   

% Expand the y output values into matrix of single values 
% for the purpose of training a NN, we need to recode the labels as vectors 
% containing only values 0 or 1

% it can be done using an eye() matrix of size num_labels, with vectorized 
% indexing by 'y'
% eye_matrix = eye(num_labels); -- then -- y_matrix = eye_matrix(y,:);
% or using repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels)
% or using a for loop 
% y_matrix = zeros(num_labels, m); % 10*5000
% for i = 1:m,
%   y_matrix(y(i),i) = 1;
% end
% or just by using the dummyvar -- all 4 options give same result
y_matrix = dummyvar(y);
% keyboard; % to debug

%  Compute non-regularized Cost Function using a3, y_matrix, & m (number of 
% training examples). the hypothesis h argument in the log() is a3 
% Cost should be a scalar value. Since y_matrix and a3 are both matrices, 
% we need to compute the double-sum sum(sum()).

product_1 = y_matrix .* log( a3 );
product_2 = ( 1 - y_matrix ) .* log( 1 - a3 );

cost = ( 1 / m) * sum( sum(- product_1 - product_2 ) );

J = cost;

% ******************** SUBMIT PART 1/1 ******************** 

% Cost Regularization: Compute the regularized component of the cost 
% Theta1 & Theta2 excluding the Theta columns for the bias units, along 
% with lambda λ, and m. easiest method is to compute the regularization 
% terms separately, then add them to the unregularized cost from part 1/1.

% add lambda term explicity excluding bias term theta index 0. we should
% not regularise theta terms corresponding to bias thats 1st column of each
% theta matrix
% we can set theta1(1) & theta2(1) to zero. Since we already calculated 
% a3 (h theta) or use the following option:
theta1_exclude_0_index = Theta1( :, 2 : end );
theta2_exclude_0_index = Theta2( :, 2 : end );

% calculating the double sum of the squares of theta
theta_sum = sum( sum ( theta1_exclude_0_index .^ 2 )) ...
                + sum( sum ( theta2_exclude_0_index .^ 2 ));
% scaling the term by (lambda / (2 * m))
cost_reg_term = ( lambda / ( 2 * m ) ) * theta_sum;
% add unregularized and regularized cost terms together to get J 
J = cost + cost_reg_term;


% ******************** SUBMIT PART 1/2 ******************** 
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
    
% implement backpropagation algorithm to compute the gradients for the 
% parameters for the (unregularized) neural network using for loop 

% big delta where layer's gradient are added
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

% process one example at time 
for t = 1:m
    % Set input layer's values to the t-th training example 
    % Perform a feedforward pass, computing the activations z2 a2 z3 a3 
    % for layers 2 and 3. ensuring a1 a2 also include the bias unit. 

    % Note I could have looped through the previously created activation a1, 
    % a2, z1 & z2 for better performance but instead i created new variables 
    % prefixed with t_ just to keep the 2 parts separate

    % input layer L1
    t_a1 = [1 X(t, :)]';    % 401x1
    % hidden layer L2
    t_z2 = Theta1 * t_a1;   % (25x401)*(401x1)
    t_a2 = sigmoid(t_z2);   % 25x1 
    t_a2 = [1 ; t_a2];      % 26x1  
    t_z3 = Theta2 * t_a2;   % (10x26) * (26x1) 
    t_a3 = sigmoid(t_z3);   % 10x1
    % dummyvar is not helpful here --> use logical array instead
    % t_y_vector = dummyvar(y(t))'; % dimension is not stable and vary with
    % values of Y
    t_y_vector = repmat([1:num_labels], 1, 1)' == repmat(y(t), 1, num_labels)'; % 10x1
       
    % calculate the errors L3 & L2 i.e the delta difference. Not L1 because 
    % there should not be an error associated with the input layer L1
    t_d3 = t_a3 - t_y_vector;
    % Exclude 1st column of Theta2 because hidden layer bias unit has no 
    % connection to the input layer - so we do not use backpropagation for it
    t_d2 = (Theta2(:, 2:end)' * t_d3) .* sigmoidGradient(t_z2); % (25x10) * (10x1) .* (25x1)
    
    % Note in production, for better performance since the cost function J 
    % already computed g(z2), it is more efficient to save this result and 
    % compute "g(z2) .* (1-g(z2))" for use during backpropagation instead 
    % of caling sigmoidGradient() which it will call sigmoid() again, which 
    % we have already computed during forward propagation.
    
    % add to big delta
    Delta1 = Delta1 + (t_d2 * t_a1'); % 25x1 * 1x401
    Delta2 = Delta2 + (t_d3 * t_a2'); % 10x1 * 1x26
    
end


% scale by 1/m and we now have an unregularised gradient 
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;



% ******************** SUBMIT PART 2 ******************** 

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Regularization: after we have verified our gradient computation for the 
% unregularized case is correct, now we can implement the gradient for the 
% regularized neural network.

% we dont regularise the bias term which is the 1st column

% for j = 0 Theta1_grad(:, 1) //excluded  
% for j >= 0 
grad1_reg_term = (lambda/m) * Theta1(:, 2:end); 
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + grad1_reg_term; 

% for j = 0 Theta2_grad(:, 1) //excluded  
% for j >= 0 
grad2_reg_term = (lambda/m) * Theta2(:, 2:end); 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + grad2_reg_term; 

% Once gradient is computed, we are able to train the neural network by 
% minimizing the cost function using an advanced optimizer such as fmincg.


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

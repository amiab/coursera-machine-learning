function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% ****  implement feedforward propagation for the neural network ****

% hypothesis computation for hidden layer & output layer

% add bias unit / add a column of 1's to X --> layer a1 
a1 = [ones(m, 1) X];
% Multiply by Theta1 --> z2
z2 = a1 * Theta1';
% apply activation function i.e. compute the sigmoid() of z2
z2_activated = sigmoid(z2);
% add bias unit / add a column of 1's --> layer a2
a2 = [ones(m, 1) z2_activated];
% Multiply by Theta2 -- z3
z3 = a2 * Theta2';
% apply activation function i.e. compute the sigmoid() of z3
z3_activated = sigmoid(z3);
% --> layer a3
a3 = z3_activated;
% use max(a3, [], 2) to return two vectors - one of the highest value for 
% each row, and one with its index. Ignore the highest values and keep the 
% vector of the indexes where the highest values were found. 
% These are your predictions.
[~, indices] = max(a3,   [], 2);
p = indices;



% =========================================================================


end

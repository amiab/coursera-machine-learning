function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% sigmoid implementation should use only element-wise operators; addition, 
% element-wise division ('./'), and the exp() function.

% exp(X) returns the exponential ex for each element in the matrix X

%Matlab
g = 1 ./ (1 + exp(-z));

% Octave we can use exp() or e
% g = 1 ./ ( 1 + e.^(-z));
  
% =============================================================

end

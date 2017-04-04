function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(theta'*X');


a = 0;
b = 0;%theta1
c = zeros(size(theta, 1) -1, 1)';%theta

for i = 1:m
    a = a+ (-y(i))*log(h(i))-(1-y(i))* log(1-h(i));
    b = b+ ((h(i)-y(i))*X(i,1));
    c = c+ ((h(i)-y(i))*X(i,2:end));
end

J = 1/m*a + lambda/2/m * sum(theta(2:end).^2);


grad = [1/m *b, 1/m*c + lambda/m*theta(2:end)'];




% =============================================================

end

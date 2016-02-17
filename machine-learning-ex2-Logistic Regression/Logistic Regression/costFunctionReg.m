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

[J_temp, grad_temp] = costFunction(theta, X, y);
% h = sigmoid(X*theta);
% J_temp = mean(-y.*log(h) - (1-y).*log(1-h));
% grad_temp = (X'*(h-y))/m;

J = J_temp + lambda/(2*m)*norm(theta(2:end))^2;
grad = grad_temp + lambda/m*theta;
grad(1) = grad_temp(1);

% =============================================================

end

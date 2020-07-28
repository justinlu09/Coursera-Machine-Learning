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

hypo = sigmoid(X * theta);

theta0 = [0; theta([2:length(theta)])];
in = log(hypo') * (-y) - log(1-hypo') * (1-y);

J = 1/m * in + (lambda/(2*m)) * sum(theta0 .^2);
grad = 1/m * sum((hypo - y) .* X) + (lambda/m) * theta0';


% =============================================================

end

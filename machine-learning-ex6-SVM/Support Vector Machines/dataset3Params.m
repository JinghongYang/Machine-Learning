function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C0 = [.1 1 10];
sigma0 = [0.05 0.1 0.2 0.4];

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

error = zeros(length(C0),length(sigma0));

for k = 1:length(C0)
    for l = 1:length(sigma0)
        model= svmTrain(X, y, C0(k), @(x1, x2) gaussianKernel(x1, x2, sigma0(l)));
        predictions = svmPredict(model, Xval);
        error(k,l) = mean(double(predictions ~= yval));
    end
end

% ind2sub(size(A),5)
[Cind,sigmaind] = find(error==min(error(:)));
C = C0(Cind);
sigma = sigma0(sigmaind);
    
    
    % =========================================================================
    
end

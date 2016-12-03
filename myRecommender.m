%
%   Tested and fully functional
%   Karthik Gopalakrishnan
%   cc.gatech.edu/~karthik46
%
% A matrix-factorization approach using gradient descent to
% infer user and movie profiles using a matrix representing
% user-movie ratings from Netflix.
%
function [ U, V ] = myRecommender( rateMatrix, lowRank )
    % Please type your name here:
    name = 'Gopalakrishnan, Karthik';
    disp(name); % Do not delete this line.

    % Parameters
    maxIter = 1000; % Choose your own.
    learningRate = 0.000095; % Choose your own.
    regularizer = 0.0000099; % Choose your own.
    
    % Random initialization:
    [n1, n2] = size(rateMatrix); % rateMatrix is n1 x n2
    U = rand(n1, lowRank) / lowRank; % n1 x lowRank
    V = rand(n2, lowRank) / lowRank; % n2 x lowRank
    
    prevU = []; % declaring for later usage
    prevV = []; % declaring for later usage
    
    iter = 0;
    % Gradient Descent:
    while iter < maxIter
        if isequal(prevU, U) && isequal(prevV, V) %% have the parameters U and V converged?
            break;
        else
            prevU = U;
            prevV = V;
        end
        
        prevVt = prevV.'; % prevV transpose, must be lowRank x n2
        errorMatrix = rateMatrix - prevU*prevVt; % n1 x n2
        errorMatrix = errorMatrix .* (rateMatrix > 0); % "zero" elements in the error matrix for non-positive ratings
        U = (1 - 2*learningRate*regularizer)*prevU + 2*learningRate*errorMatrix*prevV;
        V = (1 - 2*learningRate*regularizer)*prevV + 2*learningRate*(errorMatrix.')*prevU;
        
        iter = iter + 1;
    end
        
end

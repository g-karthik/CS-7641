%
%   Tested and fully functional
%   Karthik Gopalakrishnan
%   cc.gatech.edu/~karthik46
%
%
% This function implements expectation-maximization (EM) for
% a mixture of multinomial distributions with the aim of
% clustering documents of text.
%
function [ class ] = mycluster( bow, K )
%
% Your goal of this assignment is implementing your own text clustering algo.
%
% Input:
%     bow: data set. Bag of words representation of text document as
%     described in the assignment.
%
%     K: the number of desired topics/clusters. 
%
% Output:
%     class: the assignment of each topic. The
%     assignment should be 1, 2, 3, etc. 
%
% For submission, you need to code your own implementation without using
% any existing libraries

    docCount = size(bow,1);
    vocabCount = size(bow,2);
    
    maxIterations = 300; % maximum number of iterations within which convergence must occur
    iter = 0;
    
    pi = rand([1 K]); % 1 x K
    mu = rand([vocabCount K]); % vocabCount x K
    pi = pi/sum(pi(:)); % normalize pi to ensure probability sums to 1
    mu = bsxfun(@rdivide, mu, sum(mu)); % normalize mu by column sum to ensure probability sums to 1
    
    gamma = zeros(docCount, K); % docCount x K; declaring for later usage
    
    class = zeros(docCount, 1); % docCount x 1; declaring for later usage
    prevClass = []; % declaring for later usage
    
    while iter < maxIterations
        
        if isequal(prevClass, class)
            break;
        else
            prevClass = class;
        end
        
        % E-step
        for i=1:docCount
            for c=1:K
                base = mu(:,c); % vocabCount x 1
                power = bow(i,:).'; % vocabCount x 1 by taking transpose
                powered = base.^power; % vocabCount x 1
                product = prod(powered); % 1 x 1
                gamma(i,c) = pi(c)*product;
            end
            gamma(i,:) = gamma(i,:) / sum(gamma(i,:));
        end
        
        % M-step for mu
        for j=1:vocabCount
            for c=1:K
                left = gamma(:,c); % docCount x 1
                right = bow(:,j);   % docCount x 1
                mu(j,c) = dot(left,right);
            end
        end
        
        mu = bsxfun(@rdivide, mu, sum(mu));        
%         for c=1:K
%             mu(:,c) = mu(:,c) / sum(mu(:,c));
%         end
        
        % M-step for pi
        for c=1:K
            pi(c) = sum(gamma(:,c))/docCount;
        end
        
        iter = iter + 1;
        
        % class assignment here
        [~, class] = max(gamma,[],2);
        
    end
    
%     if iter < maxIterations
%         fprintf('mycluster converged before max iterations limit\n');
%     else
%         fprintf('mycluster max iterations limit reached\n');
%     end
    
end


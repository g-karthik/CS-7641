%
%   Tested and fully functional
%   Karthik Gopalakrishnan
%   cc.gatech.edu/~karthik46
%
%
% This function implements expectation-maximization for 
% maximum likelihood estimation in order to cluster text
% documents. It returns the posterior probabilities for
% the words and documents given the cluster/topic as well
% as the topic mixture probabilities.
%
function [ W_Z, D_Z, Z ] = mycluster2( bow, K )

    docCount = size(bow,1);
    vocabCount = size(bow,2);
    
    maxIterations = 10; % maximum number of iterations within which convergence must occur
    iter = 0;
    
    % random initialization
    W_Z = rand([vocabCount K]); % vocabCount x K
    D_Z = rand([docCount K]);   % docCount x K
    Z = rand([1 K]);            % 1 x K
    
    % normalization
    W_Z = bsxfun(@rdivide, W_Z, sum(W_Z));
    D_Z = bsxfun(@rdivide, D_Z, sum(D_Z));
    Z = Z/sum(Z(:));
    
    Z_DW = zeros(K, docCount, vocabCount); % K x docCount x vocabCount
    
    while iter < maxIterations
        
        % E-step
        for i=1:K
            for j=1:docCount
                Z_DW(i,j,:) = Z(i)*D_Z(j,i)*(W_Z(:,i).');
            end
        end
        %Z_DW = Z_DW / sum(Z_DW(:));
        Z_DW = bsxfun(@rdivide, Z_DW, sum(Z_DW));
        
        % M-step for W_Z
        for j=1:K
            for i=1:vocabCount
                W_Z(i,j) = dot(bow(:,i), squeeze(Z_DW(j,:,i)).');
            end
            % W_Z(:,j) = W_Z(:,j)/sum(W_Z(:,j));
        end
        tempSumW_Z = sum(W_Z); % 1 x K
        W_Z = bsxfun(@rdivide, W_Z, tempSumW_Z);
        
        % M-step for D_Z
        for j=1:K
            for i=1:docCount
                D_Z(i,j) = dot(bow(i,:), squeeze(Z_DW(j,i,:)).');
            end
            % D_Z(:,j) = D_Z(:,j)/sum(D_Z(:,j));
        end
        % D_Z = bsxfun(@rdivide, D_Z, sum(D_Z));
        % sum(D_Z) should be same as tempSumW_Z
        D_Z = bsxfun(@rdivide, D_Z, tempSumW_Z);
        
        % M-step for Z
        Z = tempSumW_Z / sum(tempSumW_Z);
        
        iter = iter + 1;
        
    end

end


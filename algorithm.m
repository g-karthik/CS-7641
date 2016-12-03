%
%   Tested and fully functional
%   Karthik Gopalakrishnan
%   cc.gatech.edu/~karthik46
%
% An implementation of the forward-backward inference algorithm
% for hidden Markov models, specifically to predict the
% "goodness" of the economy using S&P 500 Index data.
%
function prob = algorithm(q)

% load the matrix, obtain metadata (number of weeks)
load sp500;
weekCount = size(price_move, 1);

alpha = zeros(weekCount, 2); % weekCount x 2; first column = good state, second column = bad state
% base initialization for forward algorithm: first observation
alpha(1,1) = (1-q)*0.2;
alpha(1,2) = q*0.8;

beta = zeros(weekCount, 2); % weekCount x 2; first column = good state, second column = bad state
% base initialization for backward algorithm: last observation
beta(weekCount, 1) = 1;
beta(weekCount, 2) = 1;

a = [0.8 0.2];

% forward iteration
for t=2:weekCount
    if price_move(t,1) == 1
        p = q;
    end
    if price_move(t,1) == -1
        p = 1 - q;
    end
    alpha(t,1) = p * dot(alpha(t-1,:), a);
    alpha(t,2) = (1-p) * dot(alpha(t-1,:), fliplr(a));    
end

pX = sum(alpha(weekCount,:));

% backward iteration
for t=38:-1:1
    if price_move(t+1,1) == 1
        p = q;
    end
    if price_move(t+1,1) == -1
        p = 1 - q;
    end
    beta(t,1) = dot([p 1-p].*a, beta(t+1,:));
    beta(t,2) = dot([p 1-p].*fliplr(a), beta(t+1,:));
end

prob = alpha(weekCount,1)*beta(weekCount,1)/pX;

plotProbabilities = zeros(weekCount, 1); % weekCount x 1
for t=1:weekCount
    plotProbabilities(t) = alpha(t,1)*beta(t,1)/pX;
end

figure;
plot(plotProbabilities);
xlabel('Week');
ylabel('Probability that the economy is good');

end

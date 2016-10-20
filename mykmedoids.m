%
%   Tested and fully functional
%   Karthik Gopalakrishnan
%   cc.gatech.edu/~karthik46
%
function [ class, centroid ] = mykmedoids( pixels, K )
%
% Input:
%     pixels: data set. Each row contains one data point. For image
%     dataset, it contains 3 columns, each column corresponding to Red,
%     Green, and Blue component.
%
%     K: the number of desired clusters. Too high value of K may result in
%     empty cluster error. Then, you need to reduce it.
%
% Output:
%     class: the class assignment of each data point in pixels. The
%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
%     of class should be either 1, 2, 3, 4, or 5. The output should be a
%     column vector with size(pixels, 1) elements.
%
%     centroid: the location of K centroids in your result. With images,
%     each centroid corresponds to the representative color of each
%     cluster. The output should be a matrix with K rows and
%     3 columns. The range of values should be [0, 255].
%     
%
% You may run the following line, then you can see what should be done.
% For submission, you need to code your own implementation without using
% the kmeans matlab function directly. That is, you need to comment it out.

%	[class, centroid] = kmeans(pixels, K);

    numOfDP = size(pixels,1);   % the number of data points in the image
    
    if numOfDP < K
        K = numOfDP;    % if the input number of clusters is larger than the number of data points, do this
    end
    
    % randomly pick initial cluster centers from the data points
    initClusterCenterIndices = randi([1,numOfDP], K, 1); % K x 1 matrix containing the random cluster center indices
    clusterCenters = pixels(initClusterCenterIndices,:); % K x 3 matrix containing the random cluster centers
    
    closestIndices = zeros(numOfDP,1);  % declaring numOfDP x 1 matrix; to be used later
    previousClosestIndices = []; % declaring for usage later
    
    maxIterations = 200; % maximum number of iterations within which convergence should occur; terminate otherwise
    iter = 0;
    
    while iter < maxIterations
    
        closestIndices = knnsearch(clusterCenters,pixels); % numOfDP x 1

        if isequal(previousClosestIndices,closestIndices)
            break;
        else
            previousClosestIndices = closestIndices;
        end

        sumOfClusterDP = zeros(size(clusterCenters)); % K x 3
        countOfClusterDP = zeros(K,1); % K x 1

        for i=1:numOfDP
            sumOfClusterDP(closestIndices(i),:) = sumOfClusterDP(closestIndices(i),:) + pixels(i,:);
            countOfClusterDP(closestIndices(i)) = countOfClusterDP(closestIndices(i)) + 1;
        end

        for i=1:K
            % NOTE: Instead of replacing each cluster center with the mean
            % of the cluster of data points as in k-means, the idea is to
            % replace it with the data point in the image that most
            % "closely" resembles the mean - the following lines do
            % exactly that
            newClusterCenterIndex = knnsearch(pixels, sumOfClusterDP(i,:) / countOfClusterDP(i));
            clusterCenters(i,:) = pixels(newClusterCenterIndex,:);
        end
    
        iter = iter + 1;
    end
    
    if iter < maxIterations
        fprintf('mykmedoids converged before max iterations limit\n');
    else
        fprintf('mykmedoids max iterations limit reached\n');
    end
    
    class = closestIndices;
    centroid = clusterCenters;
    
end


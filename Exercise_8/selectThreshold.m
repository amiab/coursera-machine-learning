function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

%  investigate which examples have a very high/low probability given the 
% distribution. 
% low probability examples are more likely to be the anomalies in our dataset. 
% to determine the anomalies we need to select a threshold epsilon using 
% F1 scrore on a cross validation set

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;

for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    
    % F1 score tells us how well we are doing on finding the ground truth 
    % anomalies given a certain threshold.  

    % For different values of epsilon we compute F1 score by computing number of 
    % examples in the current threshold classifies correctly and incorrectly.
    % The  score is computed using precision (prec) and recall (rec):
        
    % check if it's low probability true / false ==> considered an anomaly
    predictions = (pval < epsilon);
    
    %  the ground truth label says it's an anomaly and our algorithm 
    % correctly classified it as an anomaly.
    true_positives = sum( ( predictions == 1 ) & ( yval == 1 ) );

    % the ground truth label says it's not an anomaly, but our algorithm 
    % incorrectly classified it as an anomaly.
    false_positives = sum( ( predictions == 1 ) & ( yval == 0 ) );
    
    % the ground truth label says it's an anomaly, but our algorithm 
    % incorrectly classified it as not being anomalous.
    false_negatives = sum( ( predictions == 0 ) & ( yval == 1 ) );
    
    precision = true_positives / (true_positives + false_positives);
    
    recall = true_positives / (true_positives + false_negatives);

    F1 = (2 * precision * recall) / (precision + recall);

    % =============================================================

    if F1 > bestF1

        bestF1 = F1;
 
        bestEpsilon = epsilon;
 
    end
end

end

function [wt] = weights_general(data, res)
% function weights_general calculates weights based on prediction errors 
% and residuals.
%
% INPUTS:
%   data: table containing a 'pe' field with prediction errors
%   res: matrix of residuals, where the first column is used
%
% OUTPUT:
%   wt: vector of weights corresponding to each data point

% EXTRACT NON-ZERO PEs
X_1 = abs(data.pe(data.pe ~= 0));

% EXTRACT FIRST COLUMN OF RESIDUALS
Y_1 = res(:,1);

% SORT PEs AND GET SORTING INDICES
[sortX_1, I_1] = sort(X_1);

wtBinSize = 1000; % Size of each bin
binMean_1 = [];
binStd_1 = [];

% CALCULATE BIN DRIFTING MEANS AND SDs OF THE PEs
for i = 1:length(X_1)-wtBinSize
    binMean_1(i) = nanmean(sortX_1(i:i+wtBinSize-1));
    binStd_1(i) = nanstd(Y_1(I_1(i:i+wtBinSize-1)));
end

% ADJUST VALUE FOR NON-ZERO BINS
% manually set all of the zero bins to the same value...
% when diff is negative, that means that the preceding val is 0
probVals_1=unique(binMean_1(~(diff(binMean_1)>0))); 
for i = 1:length(probVals_1)
    selZero_1 = binMean_1 == probVals_1(i);
    binMean_1 = [probVals_1(i) binMean_1(~selZero_1)];
    binStd_1 = [mean(binStd_1(selZero_1)), binStd_1(~selZero_1)];
end

% INITIALIZE WEIGHT VECTOR
wt = nan(size(X_1));

% ASSIGN WEIGHTS THROUGH INTERPOLATION
for i = 1:length(X_1)
    if abs(X_1(i)) > max(binMean_1)
        wt(i) = binStd_1(end);
    elseif abs(X_1(i)) < min(binMean_1)
        wt(i) = binStd_1(1);
    else
        wt(i) = interp1(binMean_1', binStd_1', abs(X_1(i)), 'linear');
    end
end
end
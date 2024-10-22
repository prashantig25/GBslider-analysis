function [wt] = weights_general(data, res)
% Todo: should be documented and cleaned a bit

%sel=isfinite(reg_rew.Var2)&isfinite(res_rew);
% reg_rew = 1;
X_1=abs(data.pe(data.pe ~= 0)); % PE

% Y_1 = reshape(res_wo_pe0,29700,1);
Y_1 = res(:,1);

[sortX_1, I_1] = sort(X_1); % sorting it

wtBinSize = 1000; % ??
binMean_1 = [];
binStd_1 = [];
% you bin the data to get drifting means and variance because you can't get
% variance for a single data point.
for i =1:length(X_1)-wtBinSize
    binMean_1(i)=nanmean(sortX_1(i:i+wtBinSize-1)); % finding bin mean
    binStd_1(i) =nanstd(Y_1(I_1(i:i+wtBinSize-1))); % finding bin std
end

% manually set all of the zero bins to the same value...
probVals_1=unique(binMean_1(~(diff(binMean_1)>0))); % when diff is negative,
% that means that the preceding val is 0
for i = 1:length(probVals_1)
    selZero_1=binMean_1==probVals_1(i);
    binMean_1= [probVals_1(i) binMean_1(~selZero_1)]; % ??
    binStd_1 = [mean(binStd_1(selZero_1)), binStd_1(~selZero_1)];
end

% interpolate and extrapolate values for each trial...
% hold on
% figure
% subplot(3,1,3)
% plot(binMean, binStd_1, 'r')
wt=nan(size(X_1));

for i =1:length(X_1)
    if abs(X_1(i))>max(binMean_1)
        wt(i)=binStd_1(end);
    elseif abs(X_1(i))<min(binMean_1)
        wt(i)=binStd_1(1);
    else
        wt(i)=(interp1(binMean_1', binStd_1', abs(X_1(i)), 'linear')); % interpolate
    end
end
end
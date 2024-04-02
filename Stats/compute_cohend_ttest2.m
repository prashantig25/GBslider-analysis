function cohen_d = compute_cohend_ttest2(mean1,mean2,sd_pooled)
    %
    % function compute_cohend_ttest2 computes cohen's d value for paired t-test.
    % INPUTS:
    % mean1: mean of within-subjects group 1
    % mean2: mean of within-subjects group 2
    % sd_pooled: pooled SD
    %
    % OUTPUT:
    % cohen_d: cohen's d
    %
    cohen_d = (mean2-mean1)./sd_pooled;
end
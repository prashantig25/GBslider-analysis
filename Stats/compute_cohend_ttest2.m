function cohen_d = compute_cohend_ttest2(mean1,mean2,sd_pooled)
    
    % function compute_cohend_ttest2 computes cohen's d value for 
    % one tailed and paired t-test.
    % INPUTS:
    %   mean1: for two-tailed, mean of within-subjects group 1
    %          for one-tailed, sample mean
    %   mean2: for two-tailed, mean of within-subjects group 2
    %          for one-tailed, population mean
    %   sd_pooled: for two-tailed, pooled SD
    %              for one-tailed, sample DF
    % OUTPUT:
    %   cohen_d: cohen's d
    
    cohen_d = (mean2-mean1)./sd_pooled;
end
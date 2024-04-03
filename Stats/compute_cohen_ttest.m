function cohen_d = compute_cohen_ttest(sample_mean,population_mean,sample_std)
    
    % function compute_cohen_ttest computes cohen's d value for one-tailed t-test.
    % INPUTS:
    %   sample_mean: mean of the sample data
    %   population_mean: mean of the population
    %   sample_std: SD of the sample data
    %
    % OUTPUT:
    %   cohen_d: cohen's d
    
    cohen_d = abs((sample_mean - population_mean) / sample_std);
end
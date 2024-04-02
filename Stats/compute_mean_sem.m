function [mean_data,sem_data] = compute_mean_sem(data)
    %
    % function compute_mean_sem computes average and standard error of mean of
    % a dataset.
    %
    % INPUTS:
        % data: dataset 
    %
    % OUTPUTS:
        % mean_data: average of data
        % sem_data: standard error of mean of data
    %         
    mean_data = nanmean(data);
    sem_data = nanstd(data)./sqrt(length(data));
end
function [mean_data,sem_data] = compute_mean_sem(data)
    mean_data = nanmean(data);
    sem_data = nanstd(data)./sqrt(length(data));
end
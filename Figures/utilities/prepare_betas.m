function [mean_avg,mean_sd,coeffs_subjs] = prepare_betas(betas_all,selected_regressors,num_subjs)

    % prepare_betas computes mean, sem and prepares for bar-graphs.

    % INPUTS:
        % betas_all: betas for all participants
        % selected_regressors: for which regressors data needs to be
        % prepared
        % num_subjs: number of subjects

   % OUTPUTS:
        % mean_avg: mean across participants
        % mean_sd: SEM across participants
        % coeffs_subjs: coefficients to be plotted
        
    coeffs = betas_all(:,selected_regressors); 
    coeffs_subjs = [];
    for n = selected_regressors
        coeffs_subjs = [coeffs_subjs; betas_all(:,n)];
    end
    
    coeffs_avg = nanmean(coeffs,1);
    coeffs_sd = nanstd(coeffs,1);
    coeffs_sem = coeffs_sd./sqrt(num_subjs);
    
    mean_avg_subjs = coeffs_avg.';
    mean_avg = mean_avg_subjs;
    
    mean_sd_subjs = coeffs_sem.';
    mean_sd = mean_sd_subjs;
end
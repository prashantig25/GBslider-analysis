clc
clearvars

AICBIC_tbl = importdata('AICBIC_integratedPerceptual_sigma_basicRLconfirm.mat');

currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)'; % to which directory one must save in
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    disp('Current directory is already the desired path. No need to run createSavePaths.');
    desiredPath = currentDir;
else
    % Call the function to create the desired path
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, filesep, "saved_figures",filesep,"main");
mkdir(save_dir)
[~,high_PU,mid_PU,low_PU,~,~,darkblue_muted,~,~,~,~,light_gray,binned_dots,barface_green,...
    reg_color,~,~,~,~] = colors_rgb(); % colors

% INITIALISE VARS
betas_all = importdata(fullfile("Data",filesep,"LR analyses",filesep,"betas_signed_wo_rewunc_obj.mat")); % participant betas
base_dir = strcat(desiredPath, filesep, 'Data');
ecoperf = importdata(strcat(base_dir, filesep, 'descriptive data', filesep, 'main study', filesep, 'ecoperf.mat')); % mean economic performance
esterror = importdata(strcat(base_dir, filesep, 'descriptive data', filesep, 'main study', filesep, 'esterror.mat')); % mean estimation error

data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);

% Initialize arrays to store results
ecoperf = NaN(length(uniqueID), 1);
esterror = NaN(length(uniqueID), 1);

% Calculate for each subject
for i = 1:length(uniqueID)
    subj_data = data(data.ID == uniqueID(i) & data.choice_cond == 2, :);
    
    ecoperf(i) = nanmean(subj_data.ecoperf);
    esterror(i) = nanmean(subj_data.est_error);
end

% Function to bin data and calculate means and SEMs
function [bin_centers, bin_means, bin_sems] = binData(x_data, y_data, n_bins)
    % Remove NaN values
    valid_idx = ~isnan(x_data) & ~isnan(y_data);
    x_clean = x_data(valid_idx);
    y_clean = y_data(valid_idx);
    
    % Create equally sized bins based on data distribution
    n_per_bin = floor(length(x_clean) / n_bins);
    [x_sorted, sort_idx] = sort(x_clean);
    y_sorted = y_clean(sort_idx);
    
    bin_centers = zeros(n_bins, 1);
    bin_means = zeros(n_bins, 1);
    bin_sems = zeros(n_bins, 1);
    
    for i = 1:n_bins
        if i < n_bins
            start_idx = (i-1) * n_per_bin + 1;
            end_idx = i * n_per_bin;
        else
            % Last bin gets remaining data points
            start_idx = (i-1) * n_per_bin + 1;
            end_idx = length(x_sorted);
        end
        
        bin_x = x_sorted(start_idx:end_idx);
        bin_y = y_sorted(start_idx:end_idx);
        
        bin_centers(i) = mean(bin_x);
        bin_means(i) = mean(bin_y);
        bin_sems(i) = std(bin_y) / sqrt(length(bin_y));
    end
end

% First figure - Estimation Error vs Delta BIC (binned)
figure
hold on

[bin_centers_est, bin_means_est, bin_sems_est] = binData(esterror, AICBIC_tbl.delta_BIC_basicRL, 5);

% Plot binned data with error bars
errorbar(bin_centers_est, bin_means_est, bin_sems_est, 'o', ...
    'MarkerSize', 8, 'MarkerFaceColor', high_PU, 'MarkerEdgeColor', 'k', ...
    'LineWidth', 1.5, 'Color', darkblue_muted, ...
    'CapSize', 6)

% Add trend line
lsline

% Calculate correlation and significance
[r_est, p_est] = corr(esterror, AICBIC_tbl.delta_BIC_basicRL, 'rows', 'complete','Type','Spearman');

% Create title with correlation and significance
if p_est < 0.001
    sig_str_est = 'p < 0.001';
elseif p_est < 0.01
    sig_str_est = sprintf('p < 0.01');
elseif p_est < 0.05
    sig_str_est = sprintf('p < 0.05');
else
    sig_str_est = sprintf('p = %.3f', p_est);
end

title(sprintf('r = %.3f, %s', r_est, sig_str_est), 'FontWeight', 'normal')
xlabel('Estimation error')
ylabel('Delta BIC for basic RL')
grid on
set(gca, 'Box', 'on')

% Second figure - Economic Performance vs Delta BIC (binned)
figure
hold on

[bin_centers_eco, bin_means_eco, bin_sems_eco] = binData(ecoperf, AICBIC_tbl.delta_BIC_basicRL, 5);

% Plot binned data with error bars
errorbar(bin_centers_eco, bin_means_eco, bin_sems_eco, 'o', ...
    'MarkerSize', 8, 'MarkerFaceColor', mid_PU, 'MarkerEdgeColor', 'k', ...
    'LineWidth', 1.5, 'Color', darkblue_muted, ...
    'CapSize', 6)

% Add trend line
lsline

% Calculate correlation and significance
[r_eco, p_eco] = corr(ecoperf, AICBIC_tbl.delta_BIC_basicRL, 'rows', 'complete','Type','Spearman');

% Create title with correlation and significance
if p_eco < 0.001
    sig_str_eco = 'p < 0.001';
elseif p_eco < 0.01
    sig_str_eco = sprintf('p < 0.01');
elseif p_eco < 0.05
    sig_str_eco = sprintf('p < 0.05');
else
    sig_str_eco = sprintf('p = %.3f', p_eco);
end

title(sprintf('r = %.3f, %s', r_eco, sig_str_eco), 'FontWeight', 'normal')
xlabel('Economic Performance')
ylabel('Delta BIC for basic RL')
grid on
set(gca, 'Box', 'on')

% Third figure - Absolute Adaptive LR vs Delta BIC (binned)
figure
hold on

% Use absolute values for adaptive LR
abs_adaptive_lr = abs(betas_all(:,2));
[bin_centers_adapt, bin_means_adapt, bin_sems_adapt] = binData(abs_adaptive_lr, AICBIC_tbl.delta_BIC_basicRL, 5);

% Plot binned data with error bars
errorbar(bin_centers_adapt, bin_means_adapt, bin_sems_adapt, 'o', ...
    'MarkerSize', 8, 'MarkerFaceColor', barface_green, 'MarkerEdgeColor', 'k', ...
    'LineWidth', 1.5, 'Color', darkblue_muted, ...
    'CapSize', 6)

% Add trend line
lsline

% Calculate correlation and significance using absolute values
[r_adaptive, p_adaptive] = corr(abs_adaptive_lr, AICBIC_tbl.delta_BIC_basicRL, 'rows', 'complete','Type','Spearman');

% Create title with correlation and significance
if p_adaptive < 0.001
    sig_str_adaptive = 'p < 0.001';
elseif p_adaptive < 0.01
    sig_str_adaptive = sprintf('p < 0.01');
elseif p_adaptive < 0.05
    sig_str_adaptive = sprintf('p < 0.05');
else
    sig_str_adaptive = sprintf('p = %.3f', p_adaptive);
end

title(sprintf('r = %.3f, %s', r_adaptive, sig_str_adaptive), 'FontWeight', 'normal')
xlabel('Absolute Adaptive LR')
ylabel('Delta BIC for basic RL')
grid on
set(gca, 'Box', 'on')

% Fourth figure - Absolute Fixed LR vs Delta BIC (binned)
figure
hold on

% Use absolute values for fixed LR
abs_fixed_lr = abs(betas_all(:,1));
[bin_centers_fixed, bin_means_fixed, bin_sems_fixed] = binData(abs_fixed_lr, AICBIC_tbl.delta_BIC_basicRL, 5);

% Plot binned data with error bars
errorbar(bin_centers_fixed, bin_means_fixed, bin_sems_fixed, 'o', ...
    'MarkerSize', 8, 'MarkerFaceColor', reg_color, 'MarkerEdgeColor', 'k', ...
    'LineWidth', 1.5, 'Color', darkblue_muted, ...
    'CapSize', 6)

% Add trend line
lsline

% Calculate correlation and significance using absolute values
[r_fixed, p_fixed] = corr(abs_fixed_lr, AICBIC_tbl.delta_BIC_basicRL, 'rows', 'complete','Type','Spearman');

% Create title with correlation and significance
if p_fixed < 0.001
    sig_str_fixed = 'p < 0.001';
elseif p_fixed < 0.01
    sig_str_fixed = sprintf('p < 0.01');
elseif p_fixed < 0.05
    sig_str_fixed = sprintf('p < 0.05');
else
    sig_str_fixed = sprintf('p = %.3f', p_fixed);
end

title(sprintf('r = %.3f, %s', r_fixed, sig_str_fixed), 'FontWeight', 'normal')
xlabel('Absolute Fixed LR')
ylabel('Delta BIC for basic RL')
grid on
set(gca, 'Box', 'on')
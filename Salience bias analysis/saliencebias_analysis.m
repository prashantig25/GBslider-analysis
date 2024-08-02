clc
clearvars

% Get the current working directory
currentDir = pwd;

% CHANGE DIRECTORY ACCORDINGLY
behv_dir = strcat('Data', filesep, 'descriptive data', filesep, 'pilot study'); % "DATA\main_study";
save_dir = strcat('saved_files', filesep, 'salience bias'); %"saved_files\study2";
mkdir(save_dir);

% PREPARE DATA
data = readtable(fullfile(behv_dir,"study1.txt")); % load
all_cond = 1; % whether all conditions to be considered for regression
if all_cond == 0
    data(data.choice_cond == 3,:) = []; % remove condition, if neccesary
end

% CATEGORICALLY CODE PERCEPTUAL AND REWARD UNCERTAINTY
for i = 1:height(data) 
    if data.condition_int(i) == 1 || data.condition_int(i) == 3
        data.reward_unc(i) = 1; % reward uncertainty
    else
        data.reward_unc(i) = 0;
    end
    if data.condition_int(i) == 4 || data.condition_int(i) == 3
        data.pu(i) = 0; % perceptual uncertainty
    else
        data.pu(i) = 1;
    end
end

% FIT MODEL
num_blocks = 16; % number of blocks
subj_id = unique(data.ID); % subject IDs
mdl = 'ecoperf ~ pu + reward_unc + contrast'; % model
num_vars = 3; % number of variables
block_data = NaN(num_blocks,num_vars); % store block wise task variables and economic performance
pred_vars = {'contrast','pu','reward_unc'}; % cell array with names of predictor variables
resp_var = 'ecoperf'; % name of response variable
cat_vars = {'contrast','pu','reward_unc'}; % cell array with names of categorical variables
num_vars = 3; % number of predictor vars
res_subjs = []; % empty array to store residuals
weight_y_n = 0; % non-weighted regression
betas_all = NaN(length(subj_id),num_vars);

for i = 1:length(subj_id)
    data_subj = data(data.ID == subj_id(i),:);
    for b = 1:num_blocks % get task-based variables and economic performance for each block
        data_blocks = data_subj(data_subj.blocks == b,:);
        block_data(b,1) = unique(data_blocks.reward_unc);
        block_data(b,2) = unique(data_blocks.pu);
        block_data(b,3) = unique(data_blocks.contrast);
        block_data(b,4) = nanmean(data_blocks.ecoperf);
    end
    tbl = table(block_data(:,1),block_data(:,2),block_data(:,3),block_data(:,4), ...
        'VariableNames',{'reward_unc','pu','contrast','ecoperf'});
    [betas,rsquared,residuals,coeffs_name,lm] = linear_fit(tbl,mdl,pred_vars,resp_var, ...
        cat_vars,num_vars,weight_y_n);
    betas_all(i,:) = betas(2:end);
end

% SAVE DATA
safe_saveall(fullfile(save_dir,"betas_salience_study1.mat"),betas_all)
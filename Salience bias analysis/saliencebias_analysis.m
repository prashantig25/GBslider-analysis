% saliencebias_analysis fits a regression model to explain the influence of
% uncertainty on salience bias in choices.

clc
clearvars

% PATH STUFF

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
save_dir = strcat(desiredPath, filesep, 'Data', filesep, 'salience bias');
mkdir(save_dir);

% LOAD DATA FOR MAIN STUDY ANALYSIS

behv_dir = strcat(desiredPath, filesep, 'Data', filesep, 'descriptive data', filesep, 'main study');
data = readtable(fullfile(behv_dir,"study2.txt")); % load
num_blocks = 12; % number of blocks

% FIT MODEL

all_cond = 1; % whether all conditions to be considered for regression
subj_id = unique(data.ID); % subject IDs
mdl = 'ecoperf ~ pu + reward_unc + contrast'; % model
num_vars = 3; % number of variables
block_data = NaN(num_blocks,num_vars); % store block wise task variables and economic performance
pred_vars = {'contrast','pu','reward_unc'}; % cell array with names of predictor variables
resp_var = 'ecoperf'; % name of response variable
cat_vars = {'contrast','pu','reward_unc'}; % cell array with names of categorical variables
num_vars = 3; % number of predictor vars
weight_y_n = 0; % non-weighted regression
betas_all = modelSalienceBias(data,num_blocks, ...
   all_cond,mdl,num_vars,block_data,pred_vars,resp_var,cat_vars, ...
   weight_y_n);

% SAVE DATA

safe_saveall(fullfile(save_dir,"betas_salience_study2.mat"),betas_all)

% RUN ANALYSIS ON PILOT DATA

behv_dir = strcat(desiredPath, filesep, 'Data', filesep, 'descriptive data', filesep, 'pilot study');
data = readtable(fullfile(behv_dir,"study1.txt")); % load
num_blocks = 16; % number of blocks
betas_all = modelSalienceBias(data,num_blocks, ...
   all_cond,mdl,num_vars,block_data,pred_vars,resp_var,cat_vars, ...
   weight_y_n);
safe_saveall(fullfile(save_dir,"betas_salience_study1.mat"),betas_all)
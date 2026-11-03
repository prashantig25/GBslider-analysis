% saliencebias_analysis fits a regression model to explain the influence of
% uncertainty on salience bias in choices.

clc
clearvars

% PATH STUFF

currentDir = cd;
reqPath = 'GBslider-analysis'; % to which directory one must save in
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
subj_id = unique(data.ID); % subject IDs

mdl = 'ecoperf ~ condiff + ev_diff + ev_diff:contrast + contrast + condiff:contrast';
cat_vars = {'contrast'}; % cell array with names of categorical variables
num_vars = 5;
pred_vars = {'contrast','pu','reward_unc','mu','condiff','ev_diff'}; % cell array with names of predictor variables
resp_var = 'ecoperf'; % name of response variable
num_subjs = length(unique(data.ID));
betas_all = NaN(num_subjs,num_vars);
for n = 1:num_subjs
    tbl = data(data.ID == subj_id(n),:);
    tbl = table(tbl.con_diff_choice,tbl.mu_congruence,tbl.contrast,tbl.ecoperf ...
        ,'VariableNames',{'condiff','mu','contrast','ecoperf'});
    tbl.ev_diff = abs(tbl.mu - (1-tbl.mu));
    tbl = table(normalise_zero_one(tbl.condiff,NaN(height(nan),1)).' ...
        ,normalise_zero_one(tbl.mu,NaN(height(nan),1)).' ...
        ,tbl.contrast,tbl.ecoperf,normalise_zero_one(tbl.ev_diff,NaN(height(tbl),1)),'VariableNames',{'condiff','mu','contrast','ecoperf','ev_diff'});
    [betas,rsquared,residuals,coeffs_name,lm] = linear_fit(tbl,mdl,pred_vars,resp_var,cat_vars,num_vars,0);
    betas_all(n,:) = betas(2:end);
end

% SAVE DATA

safe_saveall(fullfile(save_dir,"betas_salienceEV.mat"),betas_all)

mdl = 'ecoperf ~ condiff + contrast';
cat_vars = {'contrast'}; % cell array with names of categorical variables
num_vars = 2;
num_subjs = length(unique(data.ID));
betas_allbin1 = NaN(num_subjs,num_vars);
betas_allbin2 = NaN(num_subjs,num_vars);

for n = [1:19,21:60,62:num_subjs]

    tbl = data(data.ID == subj_id(n),:);
    tbl.ev_diff = abs(tbl.mu - (1-tbl.mu));
    tbl.ev_bins = discretize(tbl.ev_diff,[min(tbl.ev_diff),nanmedian(tbl.ev_diff),max(tbl.ev_diff)]);
    tbl = table(normalise_zero_one(tbl.con_diff_choice,NaN(height(nan),1)).' ...
        ,normalise_zero_one(tbl.mu_congruence,NaN(height(nan),1)).' ...
        ,tbl.contrast,tbl.ecoperf,tbl.ev_bins,'VariableNames',{'condiff','mu','contrast','ecoperf','ev_bins'});
    tbl_bin1 = tbl(tbl.ev_bins == 1,:);
    [betas,rsquared,residuals,coeffs_name,lm] = linear_fit(tbl_bin1,mdl,pred_vars,resp_var,cat_vars,num_vars,0);
    betas_allbin1(n,:) = betas(2:end);

    tbl_bin1 = tbl(tbl.ev_bins == 2,:);
    tbl = table(normalise_zero_one(tbl.condiff,NaN(height(nan),1)).' ...
        ,normalise_zero_one(tbl.mu,NaN(height(nan),1)).' ...
        ,tbl.contrast,tbl.ecoperf,'VariableNames',{'condiff','mu','contrast','ecoperf'});
    [betas,rsquared,residuals,coeffs_name,lm] = linear_fit(tbl,mdl,pred_vars,resp_var,cat_vars,num_vars,0);
    betas_allbin2(n,:) = betas(2:end);
end

% SAVE DATA

safe_saveall(fullfile(save_dir,"betas_salienceEVbin1.mat"),betas_allbin1)
safe_saveall(fullfile(save_dir,"betas_salienceEVbin2.mat"),betas_allbin2)

% FIT MODEL

all_cond = 1; % whether all conditions to be considered for regression
mdl = 'ecoperf ~ pu + reward_unc + contrast'; % model
cat_vars = {'contrast','pu','reward_unc'}; % cell array with names of categorical variables
num_vars = 4; % number of variables
block_data = NaN(num_blocks,num_vars); % store block wise task variables and economic performance
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
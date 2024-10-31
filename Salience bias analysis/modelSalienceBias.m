function betas_all = modelSalienceBias(data,num_blocks, ...
   all_cond,mdl,num_vars,block_data,pred_vars,resp_var,cat_vars, ...
   weight_y_n)
% function MODELSALIENCEBIAS Fits a linear model to analyze salience bias
%
%   INPUTS:
%   data: table containing participant data
%   num_blocks: number of blocks in the experiment
%   all_cond: consider all conditions (1) or remove condition 3 (0)
%   mdl: model definition
%   num_vars: number of variables in the regression model
%   block_data: variable to store block data
%   pred_vars: predictor variables
%   resp_var: response variable
%   cat_vars: categorical variable
%   weight_y_n: weighted regression or not
%
%   OUTPUT:
%   betas_all: matrix of beta coefficients for each subject

% PREPARE DATA
if all_cond == 0
    data(data.choice_cond == 3,:) = []; % remove condition, if necessary
end

% CATEGORICALLY CODE PERCEPTUAL AND REWARD UNCERTAINTY
data.reward_unc = double(data.condition == 1 | data.condition == 3);
data.pu = double(~(data.condition == 4 | data.condition == 3));

% FIT MODEL
subj_id = unique(data.ID); % subject IDs
betas_all = NaN(length(subj_id), num_vars);

for i = 1:length(subj_id)
    data_subj = data(data.ID == subj_id(i), :);
    for b = 1:num_blocks % get task-based variables and economic performance for each block
        data_blocks = data_subj(data_subj.blocks == b, :);
        block_data(b, 1) = unique(data_blocks.reward_unc);
        block_data(b, 2) = unique(data_blocks.pu);
        block_data(b, 3) = unique(data_blocks.contrast);
        block_data(b, 4) = nanmean(data_blocks.ecoperf);
    end
    tbl = array2table(block_data, 'VariableNames', {'reward_unc', 'pu', 'contrast', 'ecoperf'});
    [betas, ~, ~, ~, ~] = linear_fit(tbl, mdl, pred_vars, resp_var, cat_vars, num_vars, weight_y_n);
    betas_all(i, :) = betas(2:end);
end
end
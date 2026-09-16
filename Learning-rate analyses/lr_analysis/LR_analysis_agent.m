% LR_analysis_agent implements the model-based learning rate analyses
% for the agent simulations. Preprocessing is done by
% preprocessAllData.m (Learning-rate analyses/preprocessing/), which
% must be run first to produce preprocessed_agentpupil0.06.mat.

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
save_dir = strcat(desiredPath, filesep, 'Data', filesep, 'model fitting', filesep, 'agent simulations for mu range', filesep, 'range mu');

% FIT THE MODEL

lr_analysis = lr_analysis_obj();
lr_analysis.filename = fullfile(save_dir, 'preprocessed_agentpupil0.06.mat');
lr_analysis.lr_mdl = 1; % run best behavioral model
lr_analysis.risk_mdl = 0; % not run model including risk regressor
lr_analysis.saliencechoice_mdl = 0; % not run model including salience choice regressor
lr_analysis.num_subjs = 99; % number of subjects
lr_analysis.absolute_analysis = 0; % not pre-process data for absolute LR analysis
lr_analysis.grouped = 0; % set to 1 if regression model needs to be fit separately for different groups of trials
lr_analysis.num_groups = 2; % number of groups for grouped regression
lr_analysis.agent = 1; % fit model to agent simulations
lr_analysis.online = 0; % dont fit model to online dataset
lr_analysis.weighted = 1; % fit weighted regression
lr_analysis.initialiseVars(); % initalize vars for modelling LR
lr_analysis.model_definition(); % define required model
[betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = lr_analysis.get_coeffs(@fitlm, @predict);
safe_saveall(fullfile(save_dir,"betas_agent_recoding_wo_rewunc.mat"),betas_all); % save betas 
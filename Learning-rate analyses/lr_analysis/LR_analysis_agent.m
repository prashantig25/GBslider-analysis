% LR_analysis_agent implements the preprocessing and model-based
% learning rate analyses for the agent simulations

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
save_dir = strcat(desiredPath, filesep, 'Data', filesep, 'LR analyses', filesep, 'agent');
mkdir(save_dir);

% SCRIPT TO RUN MODEL BASED ANALYSIS OF LEARNING RATES

preprocess_obj = preprocess_LR(); % initialise object with all required variables and functions
preprocess_obj.filename = strcat(desiredPath, filesep, "Data", filesep, "agent simulations", filesep, "data_agent.txt");
preprocess_obj.online = 0; % not running preprocessing for participants' data
preprocess_obj.agent = 1; % running preprocessing for agent simulations
preprocess_obj.num_subjs = 99; % number of subjects
preprocess_obj.initivaliseVars; % initialize all vars for preprocessing

% ADD SIMULATION IDs
simulation_ids = [1:99];
ids = [];
for s = simulation_ids
    ids = [ids; repelem(s,100,1)];
end
all_ids = repmat(ids,3,1);
preprocess_obj.data.ID = all_ids; % add simulation IDs
preprocess_obj.compute_action_dep_rew(); % compute action dependent reward
for i = 2:height(preprocess_obj.data) % compute mu and previous trial's mu
    preprocess_obj.mu_t_1(i) = preprocess_obj.flipped_mu(i-1);
    preprocess_obj.mu_t(i) = preprocess_obj.flipped_mu(i);
end
preprocess_obj.compute_state_dep_pe(); % compute state dependent PE and UP

% COMPUTE VARS FOR LINEAR FIT
preprocess_obj.compute_ru(); % reward uncertainty
preprocess_obj.compute_confirm(); % confirming outcome
preprocess_obj.removed_cond = 3; % code for the experimental condition to be removed
preprocess_obj.remove_conditions(); % remove conditions
norm_condiff = preprocess_obj.compute_normalise(abs(preprocess_obj.data.contrast_diff)); % normalised contrast difference
preprocess_obj.add_splithalf(); % add variable to calculate splithalf reliability

% ADD VARIABLES TO THE DATA TABLE
preprocess_obj.add_vars(norm_condiff,{'norm_condiff'}); % normalised contrast difference
preprocess_obj.add_vars(preprocess_obj.data.ru,'reward_unc'); % reward uncertainty
preprocess_obj.add_vars(preprocess_obj.data.confirm_rew,'pe_sign'); % confirmating outcome

% EXCLUDE TRIALS
preprocess_obj.remove_zero_pe(); % remove trials with PE = 0

% Get the current working directory
currentDir = pwd;

% SAVE PREPROCESSED FILE
safe_saveall(fullfile(save_dir,'preprocessed_agent.mat'),preprocess_obj.data);

% FIT THE MODEL

lr_analysis = lr_analysis_obj();
lr_analysis.filename = strcat(desiredPath, filesep, "Data", filesep, "LR analyses", filesep, "agent", filesep, "preprocessed_agent.mat");
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
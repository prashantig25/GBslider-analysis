% preprocessAllData runs every preprocessing pipeline built on preprocess_LR
% and saves the resulting tables to disk. Downstream scripts load these
% files directly instead of repeating the preprocessing steps themselves:
%   1) LR ANALYSIS         -> LR_analysis_preprint.m
%   2) MODEL FITTING       -> fitSlider_ALLmodels.load_fitting_data() /
%                              fitReducedModelSpace.m and related scripts
%   3) AGENT (pupil)       -> LR_analysis_agent.m
%   4) AGENT (mu range)    -> preprocessing only, no downstream fitting script

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

%% ===================================================================
%  1) LR ANALYSIS PREPROCESSING
%  -------------------------------------------------------------------
%  Produces Data/LR analyses/preprocessed_data.mat and its splithalf
%  files, loaded by LR_analysis_preprint.m.
%  ====================================================================

save_dir = strcat(desiredPath, filesep, 'Data', filesep, 'LR analyses');
mkdir(save_dir);

preprocess_obj = preprocess_LR(); % initialise object with all required variables and functions
preprocess_obj.filename = strcat(desiredPath, filesep, "Data", filesep, "descriptive data", filesep, "main study", filesep, "study2.txt"); % specify path to get the dataset
data = readtable(preprocess_obj.filename);
subjIDs = unique(data.ID);
preprocess_obj.num_subjs = length(subjIDs); % number of subjects
preprocess_obj.online = 1; % running preprocessing for participants' data
preprocess_obj.agent = 0; % not running preprocessing for agent simulations
preprocess_obj.initivaliseVars;

preprocess_obj.flip_mu(); % compute reported contingency parameter, after correcting for congruence
preprocess_obj.compute_action_dep_rew(); % compute action dependent reward
preprocess_obj.compute_mu(); % recode mu, contingent on if actual mu < 0.5 or not
preprocess_obj.compute_state_dep_pe(); % compute state dependent PE and UP
preprocess_obj.compute_ru(); % reward uncertainty
preprocess_obj.compute_confirm(); % confirming outcome
preprocess_obj.removed_cond = 3; % code for the experimental condition to be removed
preprocess_obj.remove_conditions(); % remove conditions
norm_condiff = preprocess_obj.compute_normalise(abs(preprocess_obj.data.con_diff_choice)); % normalised contrast difference
preprocess_obj.add_splithalf(); % add variable to calculate splithalf reliability
preprocess_obj.add_saliencechoice(); % add variable wrt to whether the salient choice was made on a trial

preprocess_obj.add_vars(norm_condiff,{'norm_condiff'}); % normalised contrast difference
preprocess_obj.add_vars(preprocess_obj.data.ru,'reward_unc'); % reward uncertainty
preprocess_obj.add_vars(preprocess_obj.data.confirm_rew,'pe_sign'); % confirmating outcome

preprocess_obj.remove_zero_pe(); % remove trials with PE = 0

safe_saveall(fullfile(save_dir,'preprocessed_data.mat'),preprocess_obj.data);

data = importdata(fullfile(save_dir,'preprocessed_data.mat'));
safe_saveall(fullfile(save_dir,'preprocessed_subj_split1.mat'),data(data.splithalf == 1,:));
safe_saveall(fullfile(save_dir,'preprocessed_subj_split0.mat'),data(data.splithalf == 0,:));

%% ===================================================================
%  2) MODEL FITTING PREPROCESSING
%  -------------------------------------------------------------------
%  Produces Data/model fitting/preprocessed_dataFitting.mat, loaded by
%  fitSlider_ALLmodels.load_fitting_data() for the RL/Bayesian model
%  fitting scripts.
%  ====================================================================

save_dir = strcat(desiredPath, filesep, 'Data', filesep, 'model fitting');
mkdir(save_dir);

preprocess_obj = preprocess_LR(); % initialise object with all required variables and functions
preprocess_obj.filename = strcat(desiredPath, filesep, "Data", filesep, "descriptive data", filesep, "main study", filesep, "study2.txt"); % specify path to get the dataset
preprocess_obj.online = 1; % running preprocessing for participants' data
preprocess_obj.agent = 0; % not running preprocessing for agent simulations
preprocess_obj.num_subjs = 98; % number of subjects
preprocess_obj.initivaliseVars;

preprocess_obj.flip_mu(); % compute reported contingency parameter, after correcting for congruence
preprocess_obj.compute_action_dep_rew(); % compute action dependent reward
preprocess_obj.compute_mu(); % recode mu, contingent on if actual mu < 0.5 or not
preprocess_obj.compute_state_dep_pe(); % compute state dependent PE and UP
preprocess_obj.compute_ru(); % reward uncertainty
preprocess_obj.compute_confirm(); % confirming outcome
preprocess_obj.add_vars(preprocess_obj.recoded_reward,'recoded_reward'); % add as a data column before filtering, so it stays row-aligned with data after remove_conditions()
preprocess_obj.removed_cond = 3; % code for the experimental condition to be removed
preprocess_obj.remove_conditions(); % remove conditions
norm_condiff = preprocess_obj.compute_normalise(abs(preprocess_obj.data.con_diff_choice)); % normalised contrast difference
preprocess_obj.add_splithalf(); % add variable to calculate splithalf reliability
preprocess_obj.add_saliencechoice(); % add variable wrt to whether the salient choice was made on a trial

preprocess_obj.add_vars(norm_condiff,{'norm_condiff'}); % normalised contrast difference
preprocess_obj.add_vars(preprocess_obj.data.ru,'reward_unc'); % reward uncertainty
preprocess_obj.add_vars(preprocess_obj.data.confirm_rew,'pe_sign'); % confirmating outcome
preprocess_obj.data.condiff_relative = (preprocess_obj.data.contrast_left - preprocess_obj.data.contrast_right) ./ 2; % relative contrast difference

safe_saveall(fullfile(save_dir,'preprocessed_dataFitting.mat'),preprocess_obj.data);

%% ===================================================================
%  3) AGENT PREPROCESSING - pupil variant
%  -------------------------------------------------------------------
%  Produces Data/model fitting/agent simulations for mu range/range mu/
%  preprocessed_agentpupil0.06.mat, loaded by LR_analysis_agent.m to
%  validate the LR regression against agent-simulated ground truth.
%  ====================================================================

save_dir = strcat(desiredPath, filesep, 'Data', filesep, 'model fitting', filesep, 'agent simulations for mu range', filesep, 'range mu');
mkdir(save_dir);

preprocess_obj = preprocess_LR(); % initialise object with all required variables and functions
preprocess_obj.filename = strcat(desiredPath, filesep, "Data", filesep, "agent simulations", filesep, "data_agentpupil0.06.txt"); % specify path to get the dataset
preprocess_obj.online = 0; % not running preprocessing for participants' data
preprocess_obj.agent = 1; % running preprocessing for agent simulations
preprocess_obj.num_subjs = 300; % number of subjects
preprocess_obj.initivaliseVars; % initialize all vars for preprocessing

% ADD SIMULATION IDs
simulation_ids = [1:300];
ids = [];
for s = simulation_ids
    ids = [ids; repelem(s,100,1)];
end
all_ids = [repmat(ids,1,1);repmat(ids,1,1)];

preprocess_obj.data.ID = all_ids; % add simulation IDs
preprocess_obj.compute_action_dep_rew(); % compute action dependent reward
for i = 2:height(preprocess_obj.data) % compute mu and previous trial's mu
    preprocess_obj.mu_t_1(i) = preprocess_obj.flipped_mu(i-1);
    preprocess_obj.mu_t(i) = preprocess_obj.flipped_mu(i);
end
preprocess_obj.compute_state_dep_pe(); % compute state dependent PE and UP

preprocess_obj.compute_ru(); % reward uncertainty
preprocess_obj.compute_confirm(); % confirming outcome
preprocess_obj.removed_cond = 3; % code for the experimental condition to be removed
preprocess_obj.remove_conditions(); % remove conditions
norm_condiff = preprocess_obj.compute_normalise(abs(preprocess_obj.data.contrast_diff)); % normalised contrast difference
preprocess_obj.add_splithalf(); % add variable to calculate splithalf reliability

preprocess_obj.add_vars(norm_condiff,{'norm_condiff'}); % normalised contrast difference
preprocess_obj.add_vars(preprocess_obj.data.ru,'reward_unc'); % reward uncertainty
preprocess_obj.add_vars(preprocess_obj.data.confirm_rew,'pe_sign'); % confirmating outcome

preprocess_obj.remove_zero_pe(); % remove trials with PE = 0

safe_saveall(fullfile(save_dir,'preprocessed_agentpupil0.06.mat'),preprocess_obj.data);

%% ===================================================================
%  4) AGENT PREPROCESSING - mu range variant
%  -------------------------------------------------------------------
%  Produces Data/model fitting/agent simulations for mu range/
%  preprocessed_agent.mat. No downstream fitting script currently
%  consumes this file.
%  ====================================================================

save_dir = strcat(desiredPath, filesep, 'Data', filesep, 'model fitting', filesep, 'agent simulations for mu range');
mkdir(save_dir);

preprocess_obj = preprocess_LR(); % initialise object with all required variables and functions
preprocess_obj.filename = strcat(desiredPath, filesep, "Data", filesep, "model fitting", filesep, "agent simulations for mu range", filesep, "data_agent.txt");
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

preprocess_obj.compute_ru(); % reward uncertainty
preprocess_obj.compute_confirm(); % confirming outcome
preprocess_obj.removed_cond = 3; % code for the experimental condition to be removed
preprocess_obj.remove_conditions(); % remove conditions
norm_condiff = preprocess_obj.compute_normalise(abs(preprocess_obj.data.contrast_diff)); % normalised contrast difference
preprocess_obj.add_splithalf(); % add variable to calculate splithalf reliability

preprocess_obj.add_vars(norm_condiff,{'norm_condiff'}); % normalised contrast difference
preprocess_obj.add_vars(preprocess_obj.data.ru,'reward_unc'); % reward uncertainty
preprocess_obj.add_vars(preprocess_obj.data.confirm_rew,'pe_sign'); % confirmating outcome

preprocess_obj.remove_zero_pe(); % remove trials with PE = 0

safe_saveall(fullfile(save_dir,'preprocessed_agent.mat'),preprocess_obj.data);

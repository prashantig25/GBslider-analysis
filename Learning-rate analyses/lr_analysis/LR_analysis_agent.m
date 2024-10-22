% Todo: summary
% - and failed to run to error

clc
clearvars 
% SCRIPT TO RUN MODEL BASED ANALYSIS OF LEARNING RATES

preprocess_obj = preprocess_LR(); % initialise object with all required variables and functions

% ADD SIMULATION IDs
simulation_ids = [1:99];
ids = [];
for s = simulation_ids
    ids = [ids; repelem(s,100,1)];
end
all_ids = repmat(ids,3,1);
preprocess_obj.data.ID = all_ids;
preprocess_obj.compute_action_dep_rew(); % compute action dependent reward
for i = 2:height(preprocess_obj.data) % compute mu and previous trial's mu
    preprocess_obj.mu_t_1(i) = preprocess_obj.flipped_mu(i-1);
    preprocess_obj.mu_t(i) = preprocess_obj.flipped_mu(i);
end
preprocess_obj.compute_state_dep_pe(); % compute state dependent PE and UP

% COMPUTE VARS FOR LINEAR FIT
preprocess_obj.compute_ru(); % reward uncertainty
preprocess_obj.compute_confirm(); % confirming outcome
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

% CHANGE DIRECTORY ACCORDINGLY
save_dir = strcat('Data', filesep, 'LR analyses' , filesep, 'agent'); 
if ~exist(save_dir)
    mkdir(save_dir);
end

% SAVE PREPROCESSED FILE
safe_saveall(fullfile(save_dir,'preprocessed_agent.xlsx'),preprocess_obj.data);

% FIT THE MODEL

lr_analysis = lr_analysis_obj();
lr_analysis.model_definition();
[betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = lr_analysis.get_coeffs(@fitlm);
safe_saveall(fullfile(save_dir,"betas_agent_recoding_wo_rewunc.mat"),betas_all); % save betas as betas_signed if running signed analysis
clc
clearvars 
% SCRIPT TO RUN MODEL BASED ANALYSIS OF LEARNING RATES

preprocess_obj = preprocess_LR(); % initialise object with all required variables and functions
preprocess_obj.flip_mu(); % compute reported contingency parameter, after correcting for congruence
preprocess_obj.compute_action_dep_rew(); % compute action dependent reward
preprocess_obj.compute_mu(); % recode mu, contingent on if actual mu < 0.5 or not
preprocess_obj.compute_state_dep_pe(); % compute state dependent PE and UP

% COMPUTE VARS FOR LINEAR FIT
preprocess_obj.compute_ru(); % reward uncertainty
preprocess_obj.compute_confirm(); % confirming outcome
preprocess_obj.remove_conditions(); % remove conditions
norm_condiff = preprocess_obj.compute_normalise(abs(preprocess_obj.data.con_diff_choice)); % normalised contrast difference
preprocess_obj.add_splithalf(); % add variable to calculate splithalf reliability
preprocess_obj.add_saliencechoice(); % add variable wrt to whether the salient choice was made on a trial

% ADD VARIABLES TO THE DATA TABLE
preprocess_obj.add_vars(norm_condiff,{'norm_condiff'}); % normalised contrast difference
preprocess_obj.add_vars(preprocess_obj.data.ru,'reward_unc'); % reward uncertainty
preprocess_obj.add_vars(preprocess_obj.data.confirm_rew,'pe_sign'); % confirmating outcome

% EXCLUDE TRIALS
preprocess_obj.remove_zero_pe(); % remove trials with PE = 0

% CHANGE DIRECTORY ACCORDINGLY
% Get the current working directory
currentDir = pwd;
save_dir = strcat('Data', filesep, 'LR analyses'); 
mkdir(save_dir);

% SAVE PREPROCESSED FILE
safe_saveall(fullfile(save_dir,'preprocessed_data.xlsx'),preprocess_obj.data);

% SAVE FILES SEPARATELY FOR GROUPED REGRESSION
grouped = 0; % 1 if files need to be saved separately for grouped regression
if grouped == 1
    data = readtable(fullfile(save_dir,'preprocessed_data.xlsx'));
    safe_saveall(fullfile(save_dir,'preprocessed_subj_split1.xlsx'),data(data.splithalf == 1,:));
    safe_saveall(fullfile(save_dir,'preprocessed_subj_split0.xlsx'),data(data.splithalf == 0,:));
end

% FIT THE MODEL
lr_analysis = lr_analysis_obj();
[betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = lr_analysis.get_coeffs(@fitlm);

% SAVE DATA 
safe_saveall(fullfile(save_dir,"betas_abs.mat"),betas_all); % save betas as betas_signed if running signed analysis
safe_saveall(fullfile(save_dir,"rsquared_wo_rewunc_obj.mat"),rsquared_full); % save r-squared values
safe_saveall(fullfile(save_dir,"posterior_up_wo_rewunc_obj.mat"),posterior_up_subjs); % save posterior updates
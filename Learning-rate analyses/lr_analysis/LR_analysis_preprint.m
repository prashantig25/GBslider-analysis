% LR_analysis_preprint implements the model-based learning rate analyses
% for the participants' data. Preprocessing is done by
% preprocessAllData.m (Learning-rate analyses/preprocessing/), which
% must be run first to produce preprocessed_data.mat and its splithalf
% files.

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
save_dir = strcat(desiredPath, filesep, 'Data', filesep, 'LR analyses');
mkdir(save_dir);

% LOAD PREPROCESSED DATA

data = importdata(fullfile(save_dir,'preprocessed_data.mat'));
subjIDs = unique(data.ID);

%% FIT ALL VERSIONS OF THE ABSOLUTE MODEL

lr_analysis = lr_analysis_obj();
% lr_analysis.pupil = 0;
lr_analysis.filename = strcat(desiredPath, filesep, "Data", filesep, "LR analyses", filesep, "preprocessed_data.mat"); % specify path to get the dataset
lr_analysis.lr_mdl = 1; % run best behavioral model
lr_analysis.risk_mdl = 0; % run model including risk regressor
lr_analysis.saliencechoice_mdl = 0; % run model including salience choice regressor
lr_analysis.num_subjs = length(subjIDs); % number of subjects (was hardcoded to 98)
lr_analysis.absolute_analysis = 1; % pre-process data for absolute LR analysis
lr_analysis.grouped = 0; % set to 1 if regression model needs to be fit separately for different groups of trials
lr_analysis.num_groups = 2; % number of groups for grouped regression
lr_analysis.agent = 0; % fit model to agent simulations
lr_analysis.online = 1; % fit model to online dataset
lr_analysis.weighted = 1;
lr_analysis.initialiseVars();
lr_analysis.model_definition();

% FIT BEST MODEL
[betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = lr_analysis.get_coeffs(@fitlm,@predict);

% FIT RISK MODEL
lr_analysis.lr_mdl = 0; % run best behavioral model
lr_analysis.risk_mdl = 1; % run model including risk regressor
lr_analysis.initialiseVars();
lr_analysis.model_definition();
[betas_abs,~,~,~,~] = lr_analysis.get_coeffs(@fitlm,@predict);

% FIT SALIENCE CHOICE MODEL
lr_analysis.risk_mdl = 0; % run model including risk regressor
lr_analysis.saliencechoice_mdl = 1; % salience choice version of model
lr_analysis.initialiseVars();
lr_analysis.model_definition();
[betas_abs_salience,~,~,~,~] = lr_analysis.get_coeffs(@fitlm,@predict);

safe_saveall(fullfile(save_dir,"betas_abs_wo_rewunc_obj.mat"),betas_all);
[~,p] = ttest(betas_all); % compute p-values
safe_saveall(fullfile(save_dir,"p_vals_abs_wo_rewunc_obj.mat"),p); % save p-values
safe_saveall(fullfile(save_dir,"betas_abs.mat"),betas_abs);
safe_saveall(fullfile(save_dir,"betas_abs_salience.mat"),betas_abs_salience);

%% FIT ALL VERSIONS OF THE SIGNED MODEL

lr_analysis = lr_analysis_obj();
lr_analysis.filename = strcat(desiredPath, filesep, "Data", filesep, "LR analyses", filesep, "preprocessed_data.mat"); % specify path to get the dataset
lr_analysis.lr_mdl = 1; % run best behavioral model
lr_analysis.risk_mdl = 0; % run model including risk regressor
lr_analysis.saliencechoice_mdl = 0; % run model including salience choice regressor
lr_analysis.num_subjs = length(subjIDs); % number of subjects
lr_analysis.absolute_analysis = 0; % pre-process data for absolute LR analysis
lr_analysis.grouped = 0; % set to 1 if regression model needs to be fit separately for different groups of trials
lr_analysis.num_groups = 2; % number of groups for grouped regression
lr_analysis.online = 1; % fit model to online dataset
lr_analysis.weighted = 1;
% lr_analysis.pupil = 0;
% lr_analysis.baseline_mdl = 0;
lr_analysis.initialiseVars();
lr_analysis.model_definition();

% FIT BEST MODEL
[betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = lr_analysis.get_coeffs(@fitlm,@predict);

% FIT RISK MODEL
lr_analysis.lr_mdl = 0; % run best behavioral model
lr_analysis.risk_mdl = 1; % run model including risk regressor
lr_analysis.initialiseVars();
lr_analysis.model_definition();
[betas_signed,~,~,~,~] = lr_analysis.get_coeffs(@fitlm,@predict);

% FIT SALIENCE CHOICE MODEL
lr_analysis.risk_mdl = 0; % run model including risk regressor
lr_analysis.saliencechoice_mdl = 1; % salience choice version of model
lr_analysis.initialiseVars();
lr_analysis.model_definition();
[betas_signed_salience,~,~,~,~] = lr_analysis.get_coeffs(@fitlm,@predict);

% SAVE
safe_saveall(fullfile(save_dir,"betas_signed_wo_rewunc_obj.mat"),betas_all); % save betas as betas_signed if running signed analysis
safe_saveall(fullfile(save_dir,"rsquared_wo_rewunc_obj.mat"),rsquared_full); % save r-squared values
safe_saveall(fullfile(save_dir,"posterior_up_wo_rewunc_obj.mat"),posterior_up_subjs); % save posterior updates
[h,p] = ttest(betas_all); % compute p-values
safe_saveall(fullfile(save_dir,"p_vals_signed_wo_rewunc_obj.mat"),p); % save p-values
safe_saveall(fullfile(save_dir,"betas_signed.mat"),betas_signed);
safe_saveall(fullfile(save_dir,"betas_signed_salience.mat"),betas_signed_salience);

%% FIT SIGNED MODEL SEPARATELY FOR THE BOTH CONDITION (CONDITION 1 ONLY)
% preprocessed_data.mat (used above) mixes conditions 1 ("Both") and 2
% ("Perceptual") together -- betas_signed_wo_rewunc_obj.mat is therefore
% an aggregate-across-both-conditions fit, not condition-specific. A
% Perceptual-only (condition 2) equivalent already exists via
% Agent/model fitting/corrRegCoeffs_deltaBIC.m, which re-preprocesses
% study2.txt down to condition 2 only before refitting (removed_cond = 3
% then removed_cond = 1, leaving condition 2) and saves
% betas_signed_wo_rewunc_obj_perceptual.mat. This section mirrors that
% same template for condition 1 instead (removed_cond = 3 then
% removed_cond = 2, leaving condition 1 = "Both"), since no "Both"-only
% equivalent existed anywhere in the repo (see conversation from
% 2026-09-22).
preprocess_obj = preprocess_LR(); % initialise object with all required variables and functions
preprocess_obj.filename = strcat(desiredPath, filesep, "Data", filesep, "descriptive data", filesep, "main study", filesep, "study2.txt"); % specify path to get the dataset
data = readtable(preprocess_obj.filename);
subjIDs = unique(data.ID);
preprocess_obj.num_subjs = length(subjIDs); % number of subjects
preprocess_obj.online = 1; % running preprocessing for participants' data
preprocess_obj.agent = 0; % not r
preprocess_obj.initivaliseVars;

% COMPUTE VARS FOR LINEAR FIT

preprocess_obj.flip_mu(); % compute reported contingency parameter, after correcting for congruence
preprocess_obj.compute_action_dep_rew(); % compute action dependent reward
preprocess_obj.compute_mu(); % recode mu, contingent on if actual mu < 0.5 or not
preprocess_obj.compute_state_dep_pe(); % compute state dependent PE and UP
preprocess_obj.compute_ru(); % reward uncertainty
preprocess_obj.compute_confirm(); % confirming outcome
preprocess_obj.removed_cond = 3; % code for the experimental condition to be removed
preprocess_obj.remove_conditions(); % remove conditions
preprocess_obj.removed_cond = 2; % code for the experimental condition to be removed (leaves condition 1 = "Both")
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

% SAVE PREPROCESSED FILE

safe_saveall(fullfile(save_dir,'preprocessed_data_both.mat'),preprocess_obj.data);

% SAVE FILES SEPARATELY FOR GROUPED REGRESSION

data = importdata(fullfile(save_dir,'preprocessed_data_both.mat'));
safe_saveall(fullfile(save_dir,'preprocessed_subj_split1_both.mat'),data(data.splithalf == 1,:));
safe_saveall(fullfile(save_dir,'preprocessed_subj_split0_both.mat'),data(data.splithalf == 0,:));

% FIT BEST MODEL

lr_analysis = lr_analysis_obj();
lr_analysis.filename = strcat(desiredPath, filesep, "Data", filesep, "LR analyses", filesep, "preprocessed_data_both.mat"); % specify path to get the dataset
lr_analysis.lr_mdl = 1; % run best behavioral model
lr_analysis.risk_mdl = 0; % run model including risk regressor
lr_analysis.saliencechoice_mdl = 0; % run model including salience choice regressor
lr_analysis.num_subjs = length(subjIDs); % number of subjects
lr_analysis.absolute_analysis = 0; % pre-process data for absolute LR analysis
lr_analysis.grouped = 0; % set to 1 if regression model needs to be fit separately for different groups of trials
lr_analysis.num_groups = 2; % number of groups for grouped regression
lr_analysis.online = 1; % fit model to online dataset
lr_analysis.weighted = 1;
lr_analysis.initialiseVars();
lr_analysis.model_definition();

[betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = lr_analysis.get_coeffs(@fitlm,@predict);

% SAVE
safe_saveall(fullfile(save_dir,"betas_signed_wo_rewunc_obj_both.mat"),betas_all); % save betas as betas_signed if running signed analysis
safe_saveall(fullfile(save_dir,"rsquared_wo_rewunc_obj_both.mat"),rsquared_full); % save r-squared values
safe_saveall(fullfile(save_dir,"posterior_up_wo_rewunc_obj_both.mat"),posterior_up_subjs); % save posterior updates
[h,p] = ttest(betas_all); % compute p-values
safe_saveall(fullfile(save_dir,"p_vals_signed_wo_rewunc_obj_both.mat"),p); % save p-values

%% FIT SIGNED MODEL SEPARATELY FOR THE PERCEPTUAL CONDITION (CONDITION 2 ONLY)
% Same template as the "Both" section above, but removed_cond = 3 then
% removed_cond = 1 (leaves condition 2 = "Perceptual"). This duplicates
% what Agent/model fitting/corrRegCoeffs_deltaBIC.m already computes and
% saves under the same filenames (betas_signed_wo_rewunc_obj_perceptual.mat
% etc.) -- added here too so both condition-specific fits live alongside
% each other in the actual LR-fitting driver script, rather than the
% Perceptual-only fit only existing inside an unrelated correlation/
% plotting script (see conversation from 2026-09-22).
preprocess_obj = preprocess_LR(); % initialise object with all required variables and functions
preprocess_obj.filename = strcat(desiredPath, filesep, "Data", filesep, "descriptive data", filesep, "main study", filesep, "study2.txt"); % specify path to get the dataset
data = readtable(preprocess_obj.filename);
subjIDs = unique(data.ID);
preprocess_obj.num_subjs = length(subjIDs); % number of subjects
preprocess_obj.online = 1; % running preprocessing for participants' data
preprocess_obj.agent = 0; % not r
preprocess_obj.initivaliseVars;

% COMPUTE VARS FOR LINEAR FIT

preprocess_obj.flip_mu(); % compute reported contingency parameter, after correcting for congruence
preprocess_obj.compute_action_dep_rew(); % compute action dependent reward
preprocess_obj.compute_mu(); % recode mu, contingent on if actual mu < 0.5 or not
preprocess_obj.compute_state_dep_pe(); % compute state dependent PE and UP
preprocess_obj.compute_ru(); % reward uncertainty
preprocess_obj.compute_confirm(); % confirming outcome
preprocess_obj.removed_cond = 3; % code for the experimental condition to be removed
preprocess_obj.remove_conditions(); % remove conditions
preprocess_obj.removed_cond = 1; % code for the experimental condition to be removed (leaves condition 2 = "Perceptual")
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

% SAVE PREPROCESSED FILE

safe_saveall(fullfile(save_dir,'preprocessed_data_perceptual.mat'),preprocess_obj.data);

% SAVE FILES SEPARATELY FOR GROUPED REGRESSION

data = importdata(fullfile(save_dir,'preprocessed_data_perceptual.mat'));
safe_saveall(fullfile(save_dir,'preprocessed_subj_split1_perceptual.mat'),data(data.splithalf == 1,:));
safe_saveall(fullfile(save_dir,'preprocessed_subj_split0_perceptual.mat'),data(data.splithalf == 0,:));

% FIT BEST MODEL

lr_analysis = lr_analysis_obj();
lr_analysis.filename = strcat(desiredPath, filesep, "Data", filesep, "LR analyses", filesep, "preprocessed_data_perceptual.mat"); % specify path to get the dataset
lr_analysis.lr_mdl = 1; % run best behavioral model
lr_analysis.risk_mdl = 0; % run model including risk regressor
lr_analysis.saliencechoice_mdl = 0; % run model including salience choice regressor
lr_analysis.num_subjs = length(subjIDs); % number of subjects
lr_analysis.absolute_analysis = 0; % pre-process data for absolute LR analysis
lr_analysis.grouped = 0; % set to 1 if regression model needs to be fit separately for different groups of trials
lr_analysis.num_groups = 2; % number of groups for grouped regression
lr_analysis.online = 1; % fit model to online dataset
lr_analysis.weighted = 1;
lr_analysis.initialiseVars();
lr_analysis.model_definition();

[betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = lr_analysis.get_coeffs(@fitlm,@predict);

% SAVE
safe_saveall(fullfile(save_dir,"betas_signed_wo_rewunc_obj_perceptual.mat"),betas_all); % save betas as betas_signed if running signed analysis
safe_saveall(fullfile(save_dir,"rsquared_wo_rewunc_obj_perceptual.mat"),rsquared_full); % save r-squared values
safe_saveall(fullfile(save_dir,"posterior_up_wo_rewunc_obj_perceptual.mat"),posterior_up_subjs); % save posterior updates
[h,p] = ttest(betas_all); % compute p-values
safe_saveall(fullfile(save_dir,"p_vals_signed_wo_rewunc_obj_perceptual.mat"),p); % save p-values

%% FIT ALL MODELS TO SPLITHALF DATA

lr_analysis = lr_analysis_obj();
lr_analysis.filename = strcat(desiredPath, filesep, "Data", filesep, "LR analyses", filesep, "preprocessed_subj_split1.mat"); % specify path to get the dataset
lr_analysis.lr_mdl = 1; % run best behavioral model
lr_analysis.risk_mdl = 0; % run model including risk regressor
lr_analysis.saliencechoice_mdl = 0; % run model including salience choice regressor
lr_analysis.num_subjs = length(subjIDs); % number of subjects
lr_analysis.absolute_analysis = 0; % pre-process data for absolute LR analysis
lr_analysis.grouped = 1; % set to 1 if regression model needs to be fit separately for different groups of trials
lr_analysis.num_groups = 2; % number of groups for grouped regression
lr_analysis.online = 1; % fit model to online dataset
% lr_analysis.pupil = 0; % to fit the model to the pupil dataset
% lr_analysis.baseline_mdl = 0;
lr_analysis.weighted = 1;
lr_analysis.initialiseVars();
lr_analysis.model_definition();

% FIT SIGNED MODEL
[betas_signed_split1,~,~,~,~] = lr_analysis.get_coeffs(@fitlm,@predict);
lr_analysis.filename = strcat(desiredPath, filesep, "Data", filesep, "LR analyses", filesep, "preprocessed_subj_split0.mat"); % specify path to get the dataset
lr_analysis.initialiseVars();
lr_analysis.model_definition();
[betas_signed_split0,~,~,~,~] = lr_analysis.get_coeffs(@fitlm,@predict);

% FIT ABSOLUTE MODEL
lr_analysis.filename = strcat(desiredPath, filesep, "Data", filesep, "LR analyses", filesep, "preprocessed_subj_split1.mat"); % specify path to get the dataset
lr_analysis.absolute_analysis = 1; % pre-process data for absolute LR analysis
lr_analysis.initialiseVars();
lr_analysis.model_definition();
[betas_abs_split1,~,~,~,~] = lr_analysis.get_coeffs(@fitlm,@predict);
lr_analysis.filename = strcat(desiredPath, filesep, "Data", filesep, "LR analyses", filesep, "preprocessed_subj_split0.mat"); % specify path to get the dataset
lr_analysis.initialiseVars();
lr_analysis.model_definition();
[betas_abs_split0,~,~,~,~] = lr_analysis.get_coeffs(@fitlm,@predict);

% SAVE
safe_saveall(fullfile(save_dir,"betas_signed_split1.mat"),betas_signed_split1);
safe_saveall(fullfile(save_dir,"betas_signed_split0.mat"),betas_signed_split0);
safe_saveall(fullfile(save_dir,"betas_abs_split1.mat"),betas_abs_split1);
safe_saveall(fullfile(save_dir,"betas_abs_split0.mat"),betas_abs_split0);
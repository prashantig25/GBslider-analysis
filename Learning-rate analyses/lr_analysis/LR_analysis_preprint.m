% LR_analysis_preprint implements the preprocessing and model-based
% learning rate analyses for the participants' data.

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

% SETTING ALL THE VARIABLES FOR THE PREPROCESSING

preprocess_obj = preprocess_LR(); % initialise object with all required variables and functions
preprocess_obj.filename = strcat(desiredPath, filesep, "Data", filesep, "descriptive data", filesep, "main study", filesep, "study2.txt"); % specify path to get the dataset
preprocess_obj.online = 1; % running preprocessing for participants' data
preprocess_obj.agent = 0; % not r
preprocess_obj.num_subjs = 98; % number of subjects
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

safe_saveall(fullfile(save_dir,'preprocessed_data.mat'),preprocess_obj.data);

% SAVE FILES SEPARATELY FOR GROUPED REGRESSION

data = importdata(fullfile(save_dir,'preprocessed_data.mat'));
safe_saveall(fullfile(save_dir,'preprocessed_subj_split1.mat'),data(data.splithalf == 1,:));
safe_saveall(fullfile(save_dir,'preprocessed_subj_split0.mat'),data(data.splithalf == 0,:));

%% FIT ALL VERSIONS OF THE ABSOLUTE MODEL

lr_analysis = lr_analysis_obj();
lr_analysis.filename = strcat(desiredPath, filesep, "Data", filesep, "LR analyses", filesep, "preprocessed_data.mat"); % specify path to get the dataset
lr_analysis.lr_mdl = 1; % run best behavioral model
lr_analysis.risk_mdl = 0; % run model including risk regressor
lr_analysis.saliencechoice_mdl = 0; % run model including salience choice regressor
lr_analysis.num_subjs = 98; % number of subjects
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
lr_analysis.num_subjs = 98; % number of subjects
lr_analysis.absolute_analysis = 0; % pre-process data for absolute LR analysis
lr_analysis.grouped = 0; % set to 1 if regression model needs to be fit separately for different groups of trials
lr_analysis.num_groups = 2; % number of groups for grouped regression
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

%% FIT ALL MODELS TO SPLITHALF DATA

lr_analysis = lr_analysis_obj();
lr_analysis.filename = strcat(desiredPath, filesep, "Data", filesep, "LR analyses", filesep, "preprocessed_subj_split1.mat"); % specify path to get the dataset
lr_analysis.lr_mdl = 1; % run best behavioral model
lr_analysis.risk_mdl = 0; % run model including risk regressor
lr_analysis.saliencechoice_mdl = 0; % run model including salience choice regressor
lr_analysis.num_subjs = 98; % number of subjects
lr_analysis.absolute_analysis = 0; % pre-process data for absolute LR analysis
lr_analysis.grouped = 1; % set to 1 if regression model needs to be fit separately for different groups of trials
lr_analysis.num_groups = 2; % number of groups for grouped regression
lr_analysis.online = 1; % fit model to online dataset
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
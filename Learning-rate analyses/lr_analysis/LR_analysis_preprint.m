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
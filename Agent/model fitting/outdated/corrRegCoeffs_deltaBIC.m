clc
clearvars

AICBIC_tbl = importdata('AICBIC_integratedBoth_sigma_basicRLconfirm.mat');

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
save_dir = fullfile(desiredPath, filesep, "saved_figures",filesep,"main");
mkdir(save_dir)
[~,high_PU,mid_PU,low_PU,~,~,darkblue_muted,~,~,~,~,light_gray,binned_dots,barface_green,...
    reg_color,~,~,~,~] = colors_rgb(); % colors

% INITIALISE VARS
betas_all = importdata(fullfile("Data",filesep,"LR analyses",filesep,"betas_signed_wo_rewunc_obj.mat")); % participant betas
base_dir = strcat(desiredPath, filesep, 'Data');
ecoperf = importdata(strcat(base_dir, filesep, 'descriptive data', filesep, 'main study', filesep, 'ecoperf.mat')); % mean economic performance
esterror = importdata(strcat(base_dir, filesep, 'descriptive data', filesep, 'main study', filesep, 'esterror.mat')); % mean estimation error

% First figure - Adaptive LR vs Delta BIC
figure
hold on
scatter(esterror, AICBIC_tbl.delta_BIC_basicRL, "filled", "o", ...
    "MarkerEdgeColor", 'k', 'MarkerFaceColor', 'b', 'LineWidth', 1, ...
    'MarkerFaceAlpha', 0.3, 'SizeData', 70)
lsline

% Calculate correlation and significance for Adaptive LR
[r_adaptive, p_adaptive] = corr(esterror, AICBIC_tbl.delta_BIC_basicRL, 'rows', 'complete','Type','Spearman');

% Create title with correlation and significance
if p_adaptive < 0.001
    sig_str_adaptive = 'p < 0.001';
elseif p_adaptive < 0.01
    sig_str_adaptive = sprintf('p < 0.01');
elseif p_adaptive < 0.05
    sig_str_adaptive = sprintf('p < 0.05');
else
    sig_str_adaptive = sprintf('p = %.3f', p_adaptive);
end

title(sprintf('r = %.3f, %s', r_adaptive, sig_str_adaptive), 'FontWeight', 'normal')
xlabel('Estimation error')
ylabel('Delta BIC for basic RL')


% First figure - Adaptive LR vs Delta BIC
figure
hold on
scatter(ecoperf, AICBIC_tbl.delta_BIC_basicRL, "filled", "o", ...
    "MarkerEdgeColor", 'k', 'MarkerFaceColor', 'b', 'LineWidth', 1, ...
    'MarkerFaceAlpha', 0.3, 'SizeData', 70)
lsline

% Calculate correlation and significance for Adaptive LR
[r_adaptive, p_adaptive] = corr(ecoperf, AICBIC_tbl.delta_BIC_basicRL, 'rows', 'complete','Type','Spearman');

% Create title with correlation and significance
if p_adaptive < 0.001
    sig_str_adaptive = 'p < 0.001';
elseif p_adaptive < 0.01
    sig_str_adaptive = sprintf('p < 0.01');
elseif p_adaptive < 0.05
    sig_str_adaptive = sprintf('p < 0.05');
else
    sig_str_adaptive = sprintf('p = %.3f', p_adaptive);
end

title(sprintf('r = %.3f, %s', r_adaptive, sig_str_adaptive), 'FontWeight', 'normal')
xlabel('Ecoperf')
ylabel('Delta BIC for basic RL')


% First figure - Adaptive LR vs Delta BIC
figure
hold on
scatter(betas_all(:,2), AICBIC_tbl.delta_BIC_basicRL, "filled", "o", ...
    "MarkerEdgeColor", 'k', 'MarkerFaceColor', 'b', 'LineWidth', 1, ...
    'MarkerFaceAlpha', 0.3, 'SizeData', 70)
lsline

% Calculate correlation and significance for Adaptive LR
[r_adaptive, p_adaptive] = corr(betas_all(:,2), AICBIC_tbl.delta_BIC_basicRL, 'rows', 'complete','Type','Spearman');

% Create title with correlation and significance
if p_adaptive < 0.001
    sig_str_adaptive = 'p < 0.001';
elseif p_adaptive < 0.01
    sig_str_adaptive = sprintf('p < 0.01');
elseif p_adaptive < 0.05
    sig_str_adaptive = sprintf('p < 0.05');
else
    sig_str_adaptive = sprintf('p = %.3f', p_adaptive);
end

title(sprintf('r = %.3f, %s', r_adaptive, sig_str_adaptive), 'FontWeight', 'normal')
xlabel('Adaptive LR')
ylabel('Delta BIC for basic RL')

% Second figure - Fixed LR vs Delta BIC
figure
hold on
scatter(betas_all(:,1), AICBIC_tbl.delta_BIC_basicRL, "filled", "o", ...
    "MarkerEdgeColor", 'k', 'MarkerFaceColor', 'b', 'LineWidth', 1, ...
    'MarkerFaceAlpha', 0.3, 'SizeData', 70)
lsline

% Calculate correlation and significance for Fixed LR
[r_fixed, p_fixed] = corr(betas_all(:,1), AICBIC_tbl.delta_BIC_basicRL, 'rows', 'complete','Type','Spearman');

% Create title with correlation and significance
if p_fixed < 0.001
    sig_str_fixed = 'p < 0.001';
elseif p_fixed < 0.01
    sig_str_fixed = sprintf('p < 0.01');
elseif p_fixed < 0.05
    sig_str_fixed = sprintf('p < 0.05');
else
    sig_str_fixed = sprintf('p = %.3f', p_fixed);
end

title(sprintf('r = %.3f, %s', r_fixed, sig_str_fixed), 'FontWeight', 'normal')
xlabel('Fixed LR')
ylabel('Delta BIC for basic RL')
%%
% LR_analysis_preprint implements the preprocessing and model-based
% learning rate analyses for the participants' data.

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
preprocess_obj.removed_cond = 1; % code for the experimental condition to be removed
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

%% FIT ALL VERSIONS OF THE SIGNED MODEL

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

% FIT BEST MODEL
[betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = lr_analysis.get_coeffs(@fitlm,@predict);

% SAVE
safe_saveall(fullfile(save_dir,"betas_signed_wo_rewunc_obj_perceptual.mat"),betas_all); % save betas as betas_signed if running signed analysis
safe_saveall(fullfile(save_dir,"rsquared_wo_rewunc_obj_perceptual.mat"),rsquared_full); % save r-squared values
safe_saveall(fullfile(save_dir,"posterior_up_wo_rewunc_obj_perceptual.mat"),posterior_up_subjs); % save posterior updates
[h,p] = ttest(betas_all); % compute p-values

%%

% First figure - Adaptive LR vs Delta BIC
figure
hold on
scatter(betas_all(:,2), AICBIC_tbl.delta_BIC_basicRL, "filled", "o", ...
    "MarkerEdgeColor", 'k', 'MarkerFaceColor', 'b', 'LineWidth', 1, ...
    'MarkerFaceAlpha', 0.3, 'SizeData', 70)
lsline

% Calculate correlation and significance for Adaptive LR
[r_adaptive, p_adaptive] = corr(betas_all(:,2), AICBIC_tbl.delta_BIC_basicRL, 'rows', 'complete','Type','Spearman');

% Create title with correlation and significance
if p_adaptive < 0.001
    sig_str_adaptive = 'p < 0.001';
elseif p_adaptive < 0.01
    sig_str_adaptive = sprintf('p < 0.01');
elseif p_adaptive < 0.05
    sig_str_adaptive = sprintf('p < 0.05');
else
    sig_str_adaptive = sprintf('p = %.3f', p_adaptive);
end

title(sprintf('r = %.3f, %s', r_adaptive, sig_str_adaptive), 'FontWeight', 'normal')
xlabel('Adaptive LR')
ylabel('Delta BIC for basic RL')

% Second figure - Fixed LR vs Delta BIC
figure
hold on
scatter(betas_all(:,1), AICBIC_tbl.delta_BIC_basicRL, "filled", "o", ...
    "MarkerEdgeColor", 'k', 'MarkerFaceColor', 'b', 'LineWidth', 1, ...
    'MarkerFaceAlpha', 0.3, 'SizeData', 70)
lsline

% Calculate correlation and significance for Fixed LR
[r_fixed, p_fixed] = corr(betas_all(:,1), AICBIC_tbl.delta_BIC_basicRL, 'rows', 'complete','Type','Spearman');

% Create title with correlation and significance
if p_fixed < 0.001
    sig_str_fixed = 'p < 0.001';
elseif p_fixed < 0.01
    sig_str_fixed = sprintf('p < 0.01');
elseif p_fixed < 0.05
    sig_str_fixed = sprintf('p < 0.05');
else
    sig_str_fixed = sprintf('p = %.3f', p_fixed);
end

title(sprintf('r = %.3f, %s', r_fixed, sig_str_fixed), 'FontWeight', 'normal')
xlabel('Fixed LR')
ylabel('Delta BIC for basic RL')
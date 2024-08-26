clc
clearvars

%% PATH STUFF

% change directory accordingly
currentDir = pwd;
save_dir = strcat('Data', filesep, 'LR analyses');
mkdir(save_dir);

%% SCRIPT TO RUN MODEL BASED ANALYSIS OF LEARNING RATES

preprocess_obj = preprocess_LR(); % initialise object with all required variables and functions
preprocess_obj.flip_mu(); % compute reported contingency parameter, after correcting for congruence
preprocess_obj.compute_action_dep_rew(); % compute action dependent reward
preprocess_obj.compute_mu(); % recode mu, contingent on if actual mu < 0.5 or not
preprocess_obj.compute_state_dep_pe(); % compute state dependent PE and UP

% COMPUTE VARS FOR LINEAR FIT
preprocess_obj.compute_ru(); % reward uncertainty
preprocess_obj.compute_confirm(); % confirming outcome
preprocess_obj.remove_conditions(); % remove conditions
if preprocess_obj.online == 1
    norm_condiff = preprocess_obj.compute_normalise(abs(preprocess_obj.data.con_diff_choice)); % normalised contrast difference
elseif preprocess_obj.agent == 1
    norm_condiff = preprocess_obj.compute_normalise(abs(preprocess_obj.data.contrast_diff)); % normalised contrast difference
end
preprocess_obj.add_splithalf(); % add variable to calculate splithalf reliability
if preprocess_obj.online == 1
    preprocess_obj.add_saliencechoice(); % add variable wrt to whether the salient choice was made on a trial
end

% ADD VARIABLES TO THE DATA TABLE
preprocess_obj.add_vars(norm_condiff,{'norm_condiff'}); % normalised contrast difference
preprocess_obj.add_vars(preprocess_obj.data.ru,'reward_unc'); % reward uncertainty
preprocess_obj.add_vars(preprocess_obj.data.confirm_rew,'pe_sign'); % confirmating outcome

% EXCLUDE TRIALS
preprocess_obj.remove_zero_pe(); % remove trials with PE = 0

% SAVE PREPROCESSED FILE
safe_saveall(fullfile(save_dir,'preprocessed_data.xlsx'),preprocess_obj.data);

% SAVE FILES SEPARATELY FOR GROUPED REGRESSION
grouped = 1; % 1 if files need to be saved separately for grouped regression
if grouped == 1
    data = readtable(fullfile(save_dir,'preprocessed_data.xlsx'));
    safe_saveall(fullfile(save_dir,'preprocessed_subj_split1.xlsx'),data(data.splithalf == 1,:));
    safe_saveall(fullfile(save_dir,'preprocessed_subj_split0.xlsx'),data(data.splithalf == 0,:));
end

%% FIT THE MODEL

lr_analysis = lr_analysis_obj();
lr_analysis.model_definition();
[betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = lr_analysis.get_coeffs(@fitlm);
%% SAVE DATA FOR VARIOUS MODELS

if lr_analysis.grouped == 0
    if lr_analysis.absolute_analysis == 1 % save absolute analyses betas
        if lr_analysis.lr_mdl == 1 % best behavioral model
            safe_saveall(fullfile(save_dir,"betas_abs_wo_rewunc_obj.mat"),betas_all);
            [h,p] = ttest(betas_all); % compute p-values
            safe_saveall(fullfile(save_dir,"p_vals_abs_wo_rewunc_obj.mat"),p); % save p-values
        elseif lr_analysis.risk_mdl == 1 % best behavioral model + risk
            safe_saveall(fullfile(save_dir,"betas_abs.mat"),betas_all);
        elseif lr_analysis.saliencechoice_mdl == 1 % salience choice version of model
            safe_saveall(fullfile(save_dir,"betas_abs_salience.mat"),betas_all);
        end
    elseif lr_analysis.absolute_analysis == 0 % save signed analyses betas
        if lr_analysis.lr_mdl == 1 % best behavioral model
            safe_saveall(fullfile(save_dir,"betas_signed_wo_rewunc_obj.mat"),betas_all); % save betas as betas_signed if running signed analysis
            safe_saveall(fullfile(save_dir,"rsquared_wo_rewunc_obj.mat"),rsquared_full); % save r-squared values
            safe_saveall(fullfile(save_dir,"posterior_up_wo_rewunc_obj.mat"),posterior_up_subjs); % save posterior updates
            [h,p] = ttest(betas_all); % compute p-values
            safe_saveall(fullfile(save_dir,"p_vals_signed_wo_rewunc_obj.mat"),p); % save p-values
        elseif lr_analysis.risk_mdl == 1 % best behavioral model + risk
            safe_saveall(fullfile(save_dir,"betas_signed.mat"),betas_all);
        elseif lr_analysis.saliencechoice_mdl == 1 % best behavioral model + risk
            safe_saveall(fullfile(save_dir,"betas_signed_salience.mat"),betas_all);
        end
    end
else
    if lr_analysis.absolute_analysis == 0
        if lr_analysis.data.splithalf(1) == 1
            safe_saveall(fullfile(save_dir,"betas_signed_split1.mat"),betas_all);
        else
            safe_saveall(fullfile(save_dir,"betas_signed_split0.mat"),betas_all);
        end
    elseif lr_analysis.absolute_analysis == 1
        if lr_analysis.data.splithalf(1) == 1
            safe_saveall(fullfile(save_dir,"betas_abs_split1.mat"),betas_all);
        else
            safe_saveall(fullfile(save_dir,"betas_abs_split0.mat"),betas_all);
        end
    end
end

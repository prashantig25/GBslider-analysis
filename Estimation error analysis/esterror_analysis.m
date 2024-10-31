% esterror_analysis computes the relationship between fixed and flexible
% LRs with the participants' belief accuracy operationalized as the mean
% estimation error

clc
clearvars

% PATH 

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
save_dir = strcat(desiredPath, filesep, 'Data', filesep, 'estimation error analysis'); 
mkdir(save_dir);

% INITIALIZE VARS

num_subjs = 98; % number of subjects
mdl = 'perf ~ pe + pe__condiff  + pe__pesign + pe__salience + pe__congruence'; % full model
mdl_pe = 'perf ~ pe__condiff + pe__salience + pe__congruence + pe__pesign'; % partial model without fixed LR
mdl_pe_condiff = 'perf ~ pe + pe__salience + pe__congruence + pe__pesign'; % partial model without BS adapted LR
mdl_pe_pesign = 'perf ~ pe + pe__condiff + pe__salience + pe__congruence'; % partial model without confirmation bias
mdl_pe_salience = 'perf ~ pe + pe__condiff + pe__congruence + pe__pesign'; % partial model without salience
mdl_pe_congruence = 'perf ~ pe + pe__condiff + pe__salience + pe__pesign'; % partial model without congruence
var_names = {'pe','pe__condiff','pe__salience','pe__congruence','pe__pesign','perf'}; % variable names

% LOAD DATA

betas_all = importdata(fullfile(desiredPath, filesep,'Data',filesep,'LR analyses' ...
        ,filesep,'betas_signed_wo_rewunc_obj.mat')); % betas from lr analysis model
data = importdata(fullfile(desiredPath, filesep, 'Data',filesep,'LR analyses',filesep,'preprocessed_data.mat')); % get choice performance


% COMPUTE ESTIMATION ERROR

id_subjs = unique(data.ID); % subject IDs
for i = 1:height(data)
    if data.choice_cond(i) == 2 % actual mu for each condition
        data.actual_mu(i) = 0.9; 
    else 
        data.actual_mu(i) = 0.7;
    end
    data.signed_est_error(i) = data.mu_congruence(i) - data.actual_mu(i); % signed
    data.abs_est_error(i) = abs(data.signed_est_error(i)); % absolute
end
for i = 1:num_subjs
    absEE(i,:) = nanmean(data.abs_est_error(data.ID == id_subjs(i)));
    signedEE(i,:) = nanmean(data.signed_est_error(data.ID == id_subjs(i)));
end

% NORMALISE ALL COEFFICIENTS

norm_betas_all = betas_all;
num_vars = 5;
for i = 1:num_vars
    norm_betas_all(:,i) = normalise_zero_one(betas_all(:,i),NaN(height(betas_all),1));
end
norm_absEE = normalise_zero_one(absEE,NaN(height(betas_all),1));
norm_signedEE = normalise_zero_one(signedEE,NaN(height(betas_all),1));

data = [norm_betas_all,norm_absEE];
data_tbl = array2table(data, 'VariableNames', var_names); % data table
safe_saveall(fullfile(save_dir,"esterror_analysis_abs_error_signed_lr.mat"),data_tbl); % save table

% FIT MODEL ABS EE and SIGNED LR (Figure 5 and S8)

esterror = lr_analysis_obj(); % linear regression object
esterror.filename = strcat(desiredPath, filesep, "Data", filesep, "estimation error analysis", filesep, "esterror_analysis_abs_error_signed_lr.mat"); % use the estimation error data fil
esterror.initialiseVars; % initialise vars
esterror.EEanalysis = 1; % fit model to conduct estimation error analysis
esterror.model_definition(mdl); % define model
esterror.weighted = 0; % non-weighted regression
esterror.weight_y_n = 0;
esterror.absolute_analysis = 0; % absolute analysis or not

% FIT FULL AND PARTIAL MODELS

[~,~,~,~,lm] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe);
[~,~,~,~,lm_pe] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe_condiff);
[~,~,~,~,lm_pe_condiff] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe_pesign);
[~,~,~,~,lm_pe_pesign] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe_salience);
[~,~,~,~,lm_pe_salience] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe_congruence);
[~,~,~,~,lm_pe_congruence] = esterror.linear_fit(data_tbl,@fitlm);

% COMPUTE PARTIAL R-SQAURE

SSE_full = lm.SSE; % r-square for full model
SSE_reduced = [lm_pe.SSE; lm_pe_condiff.SSE; lm_pe_salience.SSE; lm_pe_congruence.SSE
    lm_pe_pesign.SSE]; % r-square for partial models
partial_rsqSSE = NaN(5,1);
for i = 1:num_vars
    partial_rsqSSE(i,1) = compute_partialrsqSSE(SSE_reduced(i),SSE_full);
end

% SAVE DATA

safe_saveall(fullfile(save_dir,filesep,"lm_abs_esterror_signed_lr.mat"),lm);
safe_saveall(fullfile(save_dir,filesep,"partialrsq_abs_esterror_signed_lr.mat"),partial_rsqSSE);

% FIT MODEL TO SIGNED EE AND SIGNED LR (Figure S7)

data = [norm_betas_all,norm_signedEE];
data_tbl = array2table(data, 'VariableNames', var_names); % data table
safe_saveall(fullfile(save_dir,filesep,"esterror_analysis_signed_both.mat"),data_tbl); % save table
esterror.filename = fullfile(save_dir,filesep,"esterror_analysis_signed_both.mat"); % use the estimation error data file
esterror.initialiseVars;
esterror.model_definition(mdl);

% FIT FULL AND PARTIAL MODELS

[~,~,~,~,lm] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe);
[~,~,~,~,lm_pe] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe_condiff);
[~,~,~,~,lm_pe_condiff] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe_pesign);
[~,~,~,~,lm_pe_pesign] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe_salience);
[~,~,~,~,lm_pe_salience] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe_congruence);
[~,~,~,~,lm_pe_congruence] = esterror.linear_fit(data_tbl,@fitlm);

% COMPUTE PARTIAL R-SQAURE

SSE_full = lm.SSE; % r-square for full model
SSE_reduced = [lm_pe.SSE; lm_pe_condiff.SSE; lm_pe_salience.SSE; lm_pe_congruence.SSE
    lm_pe_pesign.SSE]; % r-square for partial models
partial_rsqSSE = NaN(5,1);
for i = 1:num_vars
    partial_rsqSSE(i,1) = compute_partialrsqSSE(SSE_reduced(i),SSE_full);
end

% SAVE DATA

safe_saveall(fullfile(save_dir,filesep,"lm_signed_esterror_signed_lr.mat"),lm);
safe_saveall(fullfile(save_dir,filesep,"partialrsq_signed_esterror_signed_lr.mat"),partial_rsqSSE);

% LOAD DATA

betas_all = importdata(fullfile(desiredPath, filesep, 'Data',filesep,'LR analyses' ...
        ,filesep,'betas_abs_wo_rewunc_obj.mat')); % betas from lr analysis model
norm_betas_all = betas_all;
for i = 1:num_vars
    norm_betas_all(:,i) = normalise_zero_one(betas_all(:,i),NaN(height(betas_all),1));
end

data = [norm_betas_all,norm_absEE];
data_tbl = array2table(data, 'VariableNames', var_names); % data table
safe_saveall(fullfile(save_dir,"esterror_analysis_abs_error_abs_lr.mat"),data_tbl); % save table

% FIT MODEL TO ABS EE AND ABS LR (Figure S9)

esterror = lr_analysis_obj(); % linear regression object
esterror.filename = strcat(desiredPath, filesep, 'Data', filesep, 'estimation error analysis', filesep, 'esterror_analysis_abs_error_abs_lr.mat'); % use the estimation error data file
esterror.initialiseVars;
esterror.EEanalysis = 1;
esterror.model_definition(mdl);
esterror.weighted = 0; % non-weighted regression
esterror.weight_y_n = 0;
esterror.absolute_analysis = 0; % absolute analysis or not

% FIT FULL AND PARTIAL MODELS

[~,~,~,~,lm] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe);
[~,~,~,~,lm_pe] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe_condiff);
[~,~,~,~,lm_pe_condiff] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe_pesign);
[~,~,~,~,lm_pe_pesign] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe_salience);
[~,~,~,~,lm_pe_salience] = esterror.linear_fit(data_tbl,@fitlm);
esterror.model_definition(mdl_pe_congruence);
[betas,rsquared,residuals,coeffs_name,lm_pe_congruence] = esterror.linear_fit(data_tbl,@fitlm);

% COMPUTE PARTIAL R-SQAURE

SSE_full = lm.SSE; % r-square for full model
SSE_reduced = [lm_pe.SSE; lm_pe_condiff.SSE; lm_pe_salience.SSE; lm_pe_congruence.SSE
    lm_pe_pesign.SSE]; % r-square for partial models
partial_rsqSSE = NaN(5,1);
for i = 1:num_vars
    partial_rsqSSE(i,1) = compute_partialrsqSSE(SSE_reduced(i),SSE_full);
end

% SAVE DATA

safe_saveall(fullfile(save_dir,filesep,"lm_abs_esterror_abs_lr.mat"),lm);
safe_saveall(fullfile(save_dir,filesep,"partialrsq_abs_esterror_abs_lr.mat"),partial_rsqSSE);
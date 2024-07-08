clc
clearvars

% INITIALISE
data_subjs = readtable("Data/LR analyses/preprocessed_data.xlsx");
id_subjs = unique(data_subjs.ID); % subject IDs
mdl = 'up ~ pe + pe:contrast_diff + pe:salience_choice + pe:congruence + pe:pe_sign'; % model definition
pred_vars = {'pe','salience','contrast_diff','congruence','condition','reward_unc','subj_est_unc' ...
        ,'reward','mu','pe_sign','pu','salience_choice'}; % cell array with names of predictor variables
resp_var = 'up'; % name of response variable
cat_vars = {'salience','congruence','condition','reward_unc','pe_sign','salience_choice'}; % cell array with names of categorical variables
num_vars = 5; % number of predictor vars
res_subjs = []; % empty array to store residuals
weight_y_n = 0; % non-weighted regression
num_subjs = 1:length(id_subjs); % number of subjects

% FIT MODEL
salience_bias = lr_analysis_obj(); % linear regression object
salience_bias.filename = 'preprocessed_data.xlsx'; % use the estimation error data file
salience_bias.mdl = mdl; % model to be fit
salience_bias.pred_vars = pred_vars; % predictor variables
salience_bias.resp_var = resp_var; % response variable
salience_bias.cat_vars = cat_vars; % categorical variables
salience_bias.num_vars = num_vars; % number of regressors
salience_bias.weighted = 1; % non-weighted regression
salience_bias.weight_y_n = 0;
salience_bias.absolute_analysis = 1; % absolute analysis or not

[betas_all,rsquared,~,coeffs_name,posterior_up_subjs] = salience_bias.get_coeffs(@fitlm);
save("betas_abs_salience.mat","betas_all") % save file
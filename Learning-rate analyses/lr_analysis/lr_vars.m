classdef lr_vars < handle
% LR_VARS superclass specifies variables for the model-based analysis of
% learning rates.
    properties
        filename = 'preprocessed_data.xlsx'; % filename with pre-processed data
        data % pre-processed data
        mdl = 'up ~ pe + pe:salience + pe:congruence + pe:pe_sign + pe:contrast_diff'; % model defininition
        pred_vars = {'pe','salience','contrast_diff','congruence','reward_unc' ...
                ,'reward','mu','pe_sign','fb_phasic','fb_tonic','patch_phasic','patch_tonic'}; % cell array with names of predictor variables
        resp_var = 'up'; % name of response variable
        cat_vars = {'salience','congruence','condition','reward_unc','pe_sign' }; % cell array with names of categorical variables
        num_vars = 5; % number of predictor variables
        res_subjs = []; % empty array to store residuals
        weight_y_n = 1; % if non-weighted regression, weight_y_n = 0
        num_subjs = 98; % number of participants
        weighted = 1; % if weighted regression needs to be run
        var_names = {'pe','contrast_diff','salience','congruence','pe_sign'}; % variable names for posterior updates
        absolute_analysis = 0; % pre-process data for absolute LR analysis
        grouped = 0; % set to 1 if regression model needs to be fit separately for different groups of trials
        num_groups = 2; % number of groups for grouped regression
        pupil = 0; % fit model to pupil dataset
    end
end
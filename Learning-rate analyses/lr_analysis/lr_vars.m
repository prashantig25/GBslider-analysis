classdef lr_vars < handle
% LR_VARS superclass specifies variables for the model-based analysis of
% learning rates.
    properties
        filename = 'preprocessed_data.xlsx'; % filename with pre-processed data
        data % pre-processed data
        mdl = 'up ~ pe + pe:contrast_diff + pe:salience +pe:congruence + pe:subj_est_unc + pe:reward_unc + pe:pe_sign'; % model defininition
        pred_vars = {'pe','salience','contrast_diff','congruence','condition','reward_unc','subj_est_unc' ...
                ,'reward','mu','pe_sign','pu'}; % cell array with names of predictor variables
        resp_var = 'up'; % name of response variable
        cat_vars = {'salience','congruence','condition','reward_unc','pe_sign' }; % cell array with names of categorical variables
        num_vars = 6; % number of predictor vars
        res_subjs = []; % empty array to store residuals
        weight_y_n = 0; % non-weighted regression
        num_subjs = 99; % number of participants
        weighted = 1; % if weighted regression needs to be run
        var_names = {'pe','contrast_diff','salience','congruence','reward_unc','subj_est_unc','pe_sign'}; % var_names for posterior updates
        absolute_analysis = 0; % pre-process data for absolute LR analysis
        h_vals % h-values after significance testing
        p_vals % p-values after significance testing
        t_vals % t-values after significance testing
    end
end
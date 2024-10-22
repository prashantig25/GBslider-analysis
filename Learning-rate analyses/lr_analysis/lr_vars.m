classdef lr_vars < handle
% LR_VARS superclass specifies variables for the model-based analysis of
% learning rates.

    properties
        filename = 'Data/LR analyses/preprocessed_subj_split1.xlsx'; % filename with pre-processed data
        data % pre-processed data
        mdl % variable to store model definition
        lr_mdl = 1 % run best behavioral model 
        risk_mdl = 0 % run model including risk regressor
        saliencechoice_mdl = 0 % run model including salience choice regressor
        num_vars % number of predictor variables
        pred_vars % cell array with names of predictor variables
        resp_var = 'up'; % name of response variable
        cat_vars % cell array with names of categorical variables
        res_subjs = []; % empty array to store residuals
        weight_y_n = 1; % if non-weighted regression, weight_y_n = 0
        num_subjs = 98; % number of participants
        weighted = 1; % if weighted regression needs to be run
        absolute_analysis = 1; % pre-process data for absolute LR analysis
        grouped = 1; % set to 1 if regression model needs to be fit separately for different groups of trials
        num_groups = 2; % number of groups for grouped regression
        pupil = 0; % fit model to pupil dataset
        agent = 0; % fit model to agent simulations
        online = 1; % fit model to online dataset
    end
end
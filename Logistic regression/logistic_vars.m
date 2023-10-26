classdef logistic_vars < handle
% LOGISTIC_VARS superclass specifies variables for the model-based analysis of
% choices.
    properties
        filename = "study2.xlsx" % name of file with behavioural data
        data % table with behavioural data
        agent = 0 % run analysis for agent or not
        mdl = ['choice ~ con_diff + contrast + reward_unc + con_diff:contrast + ' ...
            'con_diff:reward_unc + contrast:reward_unc + prev_rew'];% model definition
        abs_diff % absolute contrast difference
        con_diff % contrast difference
        reward % reward obtained 
        trials % trial number
        prev_rew % reward obtained on previous trial
        condition % experimental condition
        removed_cond = 3 % condition to be excluded
        ru % if reward uncertainty is low
        pred_vars = {'contrast','condition','prev_rew','con_diff','reward_unc'}; % cell array with names of predictor variables
        resp_var = 'choice'; % name of response variable
        cat_vars = {'condition','contrast','reward_unc'}; % cell array with names of categorical variables
        num_vars = 6; % number of predictor vars
        num_subjs = 98; % number of subjects
        distribution = 'binomial'; % distribution for logistic fit
        h_vals % h-values after significance testing
        p_vals % p-values after significance testing
        t_vals % t-values after significance testing
        cat_choice % categorical correct vs. incorrect
    end
end
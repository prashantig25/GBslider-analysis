classdef preprocess_vars < handle
% PREPROCESS_VARS is a superclass that specifies variables for the preprocessing of
% behavioural data, to compute regressors for model based analyses.
    properties
        filename = 'study2.xlsx'; % path of file with behavioural data
        agent = 0 % if analysis is being run for normative agent
        pupil = 0; % if analysis is being run for behavioural data from pupil study
        space = 0; % if analysis is being run for space task data
        online = 1; % if analysis is being run on online dataset
        removed_cond = 3 % experimental condition number that is to be excluded during analysis
        num_subjs = 98; % number of participants
        absolute_lr = 0; % if analysis should be fit to absolute UPs and absolute PEs 
        data % table with behavioural data
        mu % reported contingency parameter/reward probability
        flipped_mu % reported contingency parameter/reward probability, corrected for congruence ref. eq 16
        obtained_reward % task generated reward
        condition % experimental condition
        action % participant's action
        state % trial state
        recoded_reward % reward recoded contingent on a (recoded to a = 0)
        mu_t % estimated mu for current trial
        mu_t_1 % estimated mu for previous trial
    end
end
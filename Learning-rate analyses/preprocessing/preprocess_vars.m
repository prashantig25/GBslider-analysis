classdef preprocess_vars < handle
    % PREPROCESS_VARS is a superclass that specifies variables for the preprocessing of
    % behavioural data, to compute regressors for model based analyses.
    
    properties
        filename % path of file with behavioural data
        agent % if analysis is being run for normative agent
        online % if analysis is being run on online dataset
        removed_cond % experimental condition number that is to be excluded during analysis
        num_subjs % number of participants
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
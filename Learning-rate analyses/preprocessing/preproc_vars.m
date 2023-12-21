classdef preproc_vars < handle
% PREPROC_VARS IS A superclass specifies variables for the preprocessing of
% behavioural data, to compute regressors for model based analyses.
    properties
        filename = 'data_subjs.xlsx'; % name of file with behavioural data
        agent = 0 % if analysis is being run for normative agent
        removed_cond = 3 % experimental condition number that is to be excluded during analysis
        num_subjs = 99; % number of participants
        data % table with behavioural data
        mu % reported contingency parameter
        flipped_mu % reported contingency parameter, corrected for congruence
        incorr_mu % reported contingency parameter, for the less rewarding option
        correct_choice % whether correct choice was chosen or not
        mu_pe % reported contingency parameter to compute PEs
        obtained_reward % task generated reward
        pe % predicition error
        up % update
        subjest % subjective estimation uncertainty
        expected_reward 
        condition % experimental condition
        ru % if reward uncertainty was low
        pe_sign % sign of predicition error
    end
end
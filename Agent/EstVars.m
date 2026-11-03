classdef EstVars < handle
    % This class specifies the estimation parameters

    properties
        T = 25  % number of trials
        B = 12 % number of blocks
        n_sim = NaN % number of simulations
        n_sp = 1  % number of starting points
        n_ker = 4  % number of kernels during parallelization
        experiment = 1  % current experiment
        agent = 1  % current agent-based computational model
        s_bnds = [0.01, 0.1]  % estimation boundaries for sigma parameter
        b_bnds = [0.01, 20.0]  % estimation boundaries for beta parameter
        a_bnds = [0, 1]  % estimation boundaries of alpha parameter
        l_bnds = [0, 1]  % estimation boundaries of lambda parameter
        s_fixedsp = 0.02  % fixed starting point of sigma parameter
        b_fixedsp = 5.0  % fixed starting point of beta parameter
        l_fixedsp = 0.5  % fixed starting point of lambda parameter
        a_fixedsp = 0.2  % fixed starting point of alpha parameter
        rand_sp = 0  % indicate if you want to use random starting points
        id = NaN  % ID of current participant
        real_data = 1  % indicate if estimation is based on real data
        est_sigma = 0  % indicate if sigma parameter is estimated
    end
end
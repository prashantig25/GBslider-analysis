function subj_est = compute_subjest(mu)

    % SUBJ_EST computes single trial subjective estimation uncertainty.
    % INPUT:
        % mu = reported contingency parameter
    % OUTPUT:
        % subj_est = subjective estimation uncertainty for each trial
        
    subj_est = mu.*(1-mu);
end
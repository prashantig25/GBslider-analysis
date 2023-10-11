function [flipped_mu,incorr_mu,mu_pe] = get_rew_mu(mu,data,flipped_mu,incorr_mu,correct_choice,mu_pe,agent)

    % GET_REW_MU recodes reported vontingency parameter (mu) for 
    % incongruent blocks, incorrect option and mu to compute 
    % prediction error.
    % INPUTS:
        % mu = mu reported by participant
        % data = data from all participants
        % flipped_mu = initialized array to store congruence corrected mu
        % incorr_mu = initialized array to store mu for less rewarding option
        % correct_choice = if rewarding option was chosen
        % mu_pe = initialized array for action dependent mu for calculation of pe
        % agent = if the recoding is being done for agent
    % OUTPUTS:
        % flipped_mu = congruence corrected mu
        % incorr_mu = mu for less rewarding option
        % mu_pe = action dependent mu for pe

    
    if agent == 1 % if recoding for agent
        flipped_mu = mu;
        incorr_mu = 1-mu;
    else
        for i = 1:height(data)
            if data.congruence(i) == 0 % for incongruent blocks
                flipped_mu(i) = 1-mu(i);
            else
                flipped_mu(i) = mu(i);
            end
        end      
        for i = 1:height(data) % for less rewarding option
            if data.congruence(i) == 0
                incorr_mu(i) = mu(i);
            else
                incorr_mu(i) = 1-mu(i);
            end
        end
    end
    
    % RECODE MU TO CALCULATE PE
    for i = 2:height(data)
         if correct_choice(i) == 1 % using observed rewards
            mu_pe(i-1) = flipped_mu(i-1);
         elseif correct_choice(i) == 0 
            mu_pe(i-1) = 1-flipped_mu(i-1);
         end
    end
end
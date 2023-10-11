function [pe,up] = get_pe_up(obtained,expected,flipped_mu,incorr_mu,choice_corr)

    % GET_PE_UP computes action contingent prediction errors and updates.
    % INPUT:
        % obtained = reward obtained by participant
        % expected = previous trial's reported mu
        % flipped_mu = mu, if participant chose the more rewarding choice
        % incorr_mu = mu, if participant chose the less rewarding choice
        % choice_corr = correct choice was chosen or not
    % OUTPUT:
        % pe = prediciton error
        % up = update
    
    % INITIALISE VARS
    pe = zeros(height(obtained),1);
    up = zeros(height(obtained),1);

    % COMPUTE PE, UP FOR SINGLE TRIALS
    for t = 2:length(obtained)
        pe(t) = obtained(t) - expected(t-1);
        if choice_corr(t) == 1
            up(t) = flipped_mu(t)-flipped_mu(t-1);
        else
            up(t) = incorr_mu(t)-incorr_mu(t-1);
        end
    end
end
function nll = nll_fitSlider_RLsigma(params, mu_hat, blocks, rewards, condiff)

alpha = params(1); % fixed LR
sigma = params(3); % beta
kappa = params(2); % kappa concentration parameter
nll_trial = NaN(length(mu_hat),1);

% mu_blocks = mu_hat(blocks == b);
eps = 1e-9;
mu_hat(mu_hat <= 0) = eps;
mu_hat(mu_hat >= 1) = 1 - eps;
q_0_0 = 0.5;

for n = 1:length(mu_hat)-1

    if blocks(n+1) - blocks(n) == 1
        q_0_0 = 0.5;
    end

    % Compute belief about perceptual state based on σ
    belief_state = normcdf(condiff(n), 0, sigma);

    if condiff(n) < 0
        pi_0 = 1-belief_state;
        pi_1 = belief_state;
    else
        pi_1 = belief_state;
        pi_0 = 1-belief_state;
    end
    

    % Update Q-values based on reward
    if pi_0 >= pi_1
        q_0_0 = q_0_0 + pi_0 * alpha * (rewards(n) - q_0_0);
    else
        q_0_0 = q_0_0 + pi_1 * alpha * ((1 - rewards(n)) - q_0_0);
    end

    % For each trial, compute alpha and beta for the Beta distribution
    a = q_0_0 * kappa;
    b = (1 - q_0_0) * kappa;

    if any(a <= 0 | b <= 0)
        disp('Invalid a or b detected:');
        disp([a(:), b(:)]);
    end

    % Evaluate the probability density of the observed mu_hat
    p = betapdf(mu_hat(n), a, b);
    % hold on
    % plot(p)

    % Negative log-likelihood
    nll_trial(n) = log(p);
end
nll = -nansum(nll_trial,"all");
end

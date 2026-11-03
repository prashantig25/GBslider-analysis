function nll = nll_fitSlider_basicRL(params, mu_hat, blocks, state, rewards)

alpha = params(1); % fixed LR
% beta = params(2); % beta
kappa = params(2); % kappa concentration parameter
nll_trial = NaN(length(mu_hat),1);

% mu_blocks = mu_hat(blocks == b);
eps = 1e-9;
mu_hat(mu_hat <= 0) = eps;
mu_hat(mu_hat >= 1) = 1 - eps;
q_0_0 = 0.5;

for n = 1:length(mu_hat)-1

    % Compute choice probabilities using softmax
    % q_0_1 = 1-q_0_0;
    % q_1_0 = 1-q_0_0;
    % q_1_1 = q_0_0;
    % 
    % if state(t) == 0
    %     Qvalues = [q_0_0, q_0_1];
    % else
    %     Qvalues = [q_0_1, q_0_0];
    % end
    % p_a_t = exp(Qvalues.*beta) / sum(exp(Qvalues.*beta));

    % Avoid log(0) by setting lower bounds
    % p1 = max(p_a_t(1), 1e-10);
    % p2 = max(p_a_t(2), 1e-10);

    if blocks(n+1) - blocks(n) == 1
        q_0_0 = 0.5;
    end

    if state(n) == 0
        q_0_0 = q_0_0 + alpha * (rewards(n) - q_0_0);
    else
        q_0_0 = q_0_0 + alpha * ((1 - rewards(n)) - q_0_0);
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

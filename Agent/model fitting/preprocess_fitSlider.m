%% =================== HELPER FUNCTION FOR SUBJECT PREPROCESSING ==========
function subj = preprocess_fitSlider(data, subjID, pupil, requireMuHat)
    % requireMuHat (optional, default true) -- drop trials with NaN
    % mu_hat. Models that fit mu_hat (the RL/Bayesian slider models) need
    % this; the perceptual-choice model only uses choice/condiff/blocks,
    % so fitPerceptualChoice.m passes requireMuHat = false to keep trials
    % with a valid choice but a missing slider (mu) response.
    if nargin < 4
        requireMuHat = true;
    end
    dataSubj = data(data.ID == subjID, :);
    % if pupil == 0
        mu_hat = dataSubj.mu_congruence;
    % else
    %     mu_hat = dataSubj.;
    % end
    if requireMuHat
        valid = ~isnan(mu_hat);
        mu_hat = mu_hat(valid);
        dataSubj = dataSubj(valid, :);
    end
    subj.mu_hat = mu_hat;
    subj.blocks = dataSubj.blocks;
    subj.state = dataSubj.state;
    subj.condiff = dataSubj.condiff_relative;
    subj.choices = dataSubj.choice;
    subj.recoded_rewards = dataSubj.recoded_reward;
    subj.rewards = dataSubj.correct;
    subj.contrast = dataSubj.contrast;
    subj.dataTable = dataSubj;
end
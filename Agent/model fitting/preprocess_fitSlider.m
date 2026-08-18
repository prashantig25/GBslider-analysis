%% =================== HELPER FUNCTION FOR SUBJECT PREPROCESSING ==========
function subj = preprocess_fitSlider(data, subjID, pupil)
    dataSubj = data(data.ID == subjID, :);
    % if pupil == 0
        mu_hat = dataSubj.mu_congruence;
    % else
    %     mu_hat = dataSubj.;
    % end
    valid = ~isnan(mu_hat);
    mu_hat = mu_hat(valid);
    dataSubj = dataSubj(valid, :);
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
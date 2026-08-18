clc
clearvars
data = importdata("genModel_AgentFixedParams_mean0.1Sigma.mat");

% Find where blocks change
block_change = [1; diff(data.blocks) ~= 0];

% Create trial counter
trial_counter = cumsum(block_change);

% For each group, assign 1:N
data.trials = zeros(height(data), 1);
for i = 1:max(trial_counter)
    idx = trial_counter == i;
    data.trials(idx) = 1:sum(idx);
end

% 3. Correlation with correct choice
data.correct = (data.state == data.choice);


data.pe = NaN(height(data),1);

mask = (data.state == data.choice);

% Initialize confirm vector
confirm = NaN(size(data.rewards));

% Assign values based on mask
confirm(mask) = data.rewards(mask);
confirm(~mask) = 1 - data.rewards(~mask);
data.confirm_rew = confirm;
fprintf('Corr(correct_choice, confirm): %.3f\n', ...
    corr(data.correct, data.confirm_rew, 'rows', 'complete'));

% Recode rewards based on choice
data.rewards(data.choice == 1) = 1 - data.rewards(data.choice == 1);

% Create logical index for state == 0
state_is_zero = data.state(2:end) == 0;

% Compute prediction errors for both cases
pe_state0 = data.rewards(2:end) - data.mu_hat(1:end-1);
pe_state1 = (1 - data.rewards(2:end)) - data.mu_hat(1:end-1);

% Combine using logical indexing
data.pe(2:end) = state_is_zero .* pe_state0 + ~state_is_zero .* pe_state1;

% up = current - previous
data.up = [NaN; data.mu_hat(2:end) - data.mu_hat(1:end-1)];

data.lr = data.up./data.pe;

corr(abs(data.condiff_relative), data.confirm_rew)

data(data.pe == 0,:) = [];
data(abs(data.lr) > 5,:) = [];
data(data.trials == 1,:) = [];
% data(isnan(data.lr),:) = [];

figure
hold on
bar([nanmean(abs(data.lr(data.confirm_rew == 1,1))),nanmean(abs(data.lr(data.confirm_rew == 0,1)))])
hold on
bar([nanmean(data.lr(data.confirm_rew == 1,1)),nanmean(data.lr(data.confirm_rew == 0,1))])
xticks([1,2])
xticklabels(["Confirming","Disconfirming"])

% Define high vs low contrast groups (using median split)
median_contrast = median(abs(data.condiff_relative));
high_contrast = abs(data.condiff_relative) >= median_contrast;
low_contrast = abs(data.condiff_relative) < median_contrast;

% Calculate mean confirm_rew for each group
mean_confirm_high = nanmean(data.confirm_rew(high_contrast));
mean_confirm_low = nanmean(data.confirm_rew(low_contrast));

%%

% Bin contrast difference into 10 bins
n_bins = 10;
[~, edges, bin_idx] = histcounts(abs(data.condiff_relative), n_bins);

% Initialize arrays to store means and SEMs
lr_confirm = NaN(n_bins, 1);
lr_noconfirm = NaN(n_bins, 1);
sem_confirm = NaN(n_bins, 1);
sem_noconfirm = NaN(n_bins, 1);
bin_centers = NaN(n_bins, 1);

% Calculate mean absolute LR for each bin
for i = 1:n_bins
    bin_centers(i) = mean([edges(i), edges(i+1)]);
    
    % Confirm trials in this bin
    idx_confirm = (bin_idx == i) & (data.confirm_rew == 1);
    if sum(idx_confirm) > 0
        lr_confirm(i) = nanmean(abs(data.lr(idx_confirm)));
        sem_confirm(i) = nanstd(abs(data.lr(idx_confirm))) / sqrt(sum(idx_confirm));
    end
    
    % No confirm trials in this bin
    idx_noconfirm = (bin_idx == i) & (data.confirm_rew == 0);
    if sum(idx_noconfirm) > 0
        lr_noconfirm(i) = nanmean(abs(data.lr(idx_noconfirm)));
        sem_noconfirm(i) = nanstd(abs(data.lr(idx_noconfirm))) / sqrt(sum(idx_noconfirm));
    end
end

% Plot
figure
hold on
errorbar(bin_centers, lr_confirm, sem_confirm, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Confirm')
errorbar(bin_centers, lr_noconfirm, sem_noconfirm, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'No Confirm')
xlabel('Absolute Contrast Difference')
ylabel('Absolute Learning Rate')
title('Learning Rate by Contrast Difficulty and Confirmation')
legend('Location', 'best')
grid on

%%

% 1. Mean confirm rate
fprintf('Mean confirm_rew: %.3f\n', mean(data.confirm_rew));

% 2. Correlation with belief certainty  
data.belief_cert = abs(data.mu_hat - 0.5);
fprintf('Corr(belief_certainty, confirm): %.3f\n', ...
    corr(data.belief_cert, data.confirm_rew, 'rows', 'complete'));


% 4. Mean PE by confirm
fprintf('Mean |PE| confirm=1: %.3f\n', nanmean(abs(data.pe(data.confirm_rew==1))));
fprintf('Mean |PE| confirm=0: %.3f\n', nanmean(abs(data.pe(data.confirm_rew==0))));
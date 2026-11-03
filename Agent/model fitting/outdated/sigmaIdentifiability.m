% Sigma Identifiability Analysis
% Integrated with your existing parameter recovery setup

clc
clearvars
rng(123)

% Create synthetic data (from your script)
n_trials = 1000;
state = randi([0 1], n_trials, 1);
condiff = NaN(n_trials, 1);

% Generate condiff values depending on state
condiff(state == 0) = -0.08 + (0 - (-0.08)) .* rand(sum(state == 0), 1);
condiff(state == 1) = 0 + (0.08 - 0) .* rand(sum(state == 1), 1);

block_size = 25; % trials per block
nBlocks = n_trials / block_size;
blocks = repelem(1:nBlocks, block_size)';

% Parameters for testing
sigmas = linspace(0.001, 0.1, 10);  % 10 sigma values from 0.001 to 0.05
alpha =  0.2;  % Fixed alpha for this test
kappa = 5;    % Fixed kappa for this test
n_trials_plot = 25;  % First 25 trials for visualization

% Collect mu_hat trajectories for each sigma
mu_trajectories = zeros(n_trials, length(sigmas));

fprintf('Generating trajectories for %d different sigma values...\n', length(sigmas));
for s = 1:length(sigmas)
    rng(123);  % Fixed seed for fair comparison
    params_test = [alpha, kappa, sigmas(s)];
    
    % Call your prediction function
    [~, mu_trajectories(:,s)] = predict_allModels.predict_RLsigma(...
        params_test, blocks, state, condiff, 'sample');
end
fprintf('Done!\n\n');

% Compute correlation matrix
corr_matrix = corr(mu_trajectories);

% === Plotting Section ===
figure('Position', [100, 100, 1200, 500]);

%% ---- Subplot 1: Lower-Triangle Correlation Matrix (Symmetrical Display) ----
subplot(1, 2, 1);

% Mask upper triangle (keep lower including diagonal)
corr_lower = tril(corr_matrix);

% Create a mask for upper triangle
mask_upper = triu(true(size(corr_matrix)), 1);

% Plot lower triangle
imagesc(corr_lower);
colormap(flipud(hot));
caxis([0 1]);
axis equal tight;
hold on;

% Overlay white patch over upper triangle to make it symmetrical
[xMask, yMask] = meshgrid(1:length(sigmas));
upper_x = xMask(mask_upper);
upper_y = yMask(mask_upper);
scatter(upper_x, upper_y, 100, [1 1 1], 'filled');  % white overlay

colorbar;
xlabel('Sigma Index');
ylabel('Sigma Index');
title('Lower-Triangle Correlation Matrix of \mu Trajectories');

% Tick labels showing sigma values
tick_labels = arrayfun(@(x) sprintf('%.3f', x), sigmas, 'UniformOutput', false);
xticks(1:length(sigmas));
xticklabels(tick_labels);
xtickangle(45);
yticks(1:length(sigmas));
yticklabels(tick_labels);

% Add numeric correlation values (only lower triangle)
for i = 2:length(sigmas)
    for j = 1:i
        if i ~= j
            text(j, i, sprintf('%.2f', corr_matrix(i,j)), ...
                'HorizontalAlignment', 'center', ...
                'Color', 'white', ...
                'FontSize', 8);
        end
    end
end

grid on;
set(gca, 'GridColor', 'w', 'GridAlpha', 0.3);
hold on;

%% ---- Subplot 2: Average Within-Block μ Trajectories (25 Trials per Block) ----
subplot(1, 2, 2);
hold on;
colors = parula(length(sigmas));

% Reshape trajectories into blocks (block_size x nBlocks x nSigmas)
mu_blocks = reshape(mu_trajectories, block_size, nBlocks, length(sigmas));

% Compute mean and SEM across blocks (so each curve = avg over 40 blocks)
block_mean_traj = squeeze(mean(mu_blocks, 2));  % (25 x nSigmas)
block_sem_traj  = squeeze(std(mu_blocks, 0, 2) / sqrt(nBlocks));  % (25 x nSigmas)

% Plot each sigma with shaded error across the 25 trials
for s = 1:length(sigmas)
    x = 1:block_size;                      % 1–25 trials within block
    y = block_mean_traj(:, s);
    sem = block_sem_traj(:, s);

    % Shaded error area
    fill([x fliplr(x)], [y - sem; flipud(y + sem)]', ...
        colors(s,:), 'FaceAlpha', 0.25, 'EdgeColor', 'none');

    % Mean line
    plot(x, y, 'Color', colors(s,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('\\sigma = %.3f', sigmas(s)));
end

xlabel('Trial Number Within Block');
ylabel('Mean \mu (Belief State)');
title('Average Within-Block \mu Trajectories (Mean ± SEM Across 40 Blocks)');
legend('Location', 'eastoutside', 'FontSize', 8);
grid on;
xlim([1 block_size]);
ylim([0 1]);
hold off;

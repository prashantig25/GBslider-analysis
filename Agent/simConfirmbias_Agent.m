% MATLAB script for running 100 simulations with and without confirmation bias and plotting the average mu

clc
clearvars

% Set number of simulations
num_simulations = 100;

% Initialize storage arrays (assuming mu has a certain length - adjust as needed)
% You may need to adjust the second dimension based on your actual mu length
mu_no_bias = [];
mu_with_bias = [];

% Run simulations without confirmation bias
agent = Agent();
agent.confirmation_bias = 0;
for i = 1:num_simulations
    sim_result = task_agent_int(1,1,0,1); % task_agent_analysis=1 preserves this script's prior (hardcoded) behavior
    if i == 1
        % Initialize arrays based on actual mu length
        mu_no_bias = zeros(num_simulations, length(sim_result.mu));
        mu_with_bias = zeros(num_simulations, length(sim_result.mu));
    end
    mu_no_bias(i, :) = sim_result.mu;
    fprintf('No bias simulation %d/%d completed\n', i, num_simulations);
end
%%
% Run simulations with confirmation bias
agent = Agent();
agent.confirmation_bias = 1;
for i = 1:num_simulations
    sim_result = task_agent_int(1,1,0,1); % task_agent_analysis=1 preserves this script's prior (hardcoded) behavior
    mu_with_bias(i, :) = sim_result.mu;
    fprintf('With bias simulation %d/%d completed\n', i, num_simulations);
end

% Calculate average mu across simulations
avg_mu_no_bias = mean(mu_no_bias, 1);
avg_mu_with_bias = mean(mu_with_bias, 1);

% Calculate standard error for error bars (optional)
se_mu_no_bias = std(mu_no_bias, 0, 1) / sqrt(num_simulations);
se_mu_with_bias = std(mu_with_bias, 0, 1) / sqrt(num_simulations);

% Create the plot
figure;
hold on;

% Plot average trajectories
plot(1:length(avg_mu_no_bias), avg_mu_no_bias, 'b-', 'LineWidth', 2, 'DisplayName', 'No Confirmation Bias');
plot(1:length(avg_mu_with_bias), avg_mu_with_bias, 'r-', 'LineWidth', 2, 'DisplayName', 'With Confirmation Bias');

% Optional: Add error bars (uncomment if desired)
errorbar(1:length(avg_mu_no_bias), avg_mu_no_bias, se_mu_no_bias, 'b', 'LineWidth', 1);
errorbar(1:length(avg_mu_with_bias), avg_mu_with_bias, se_mu_with_bias, 'r', 'LineWidth', 1);

% Formatting
xlabel('Time Steps');
ylabel('Average mu');
title(sprintf('Average mu across %d simulations', num_simulations));
legend('Location', 'best');
grid on;
hold off;

% Display summary statistics
fprintf('\nSummary Statistics:\n');
fprintf('No Confirmation Bias - Final mu: %.4f ± %.4f\n', avg_mu_no_bias(end), se_mu_no_bias(end));
fprintf('With Confirmation Bias - Final mu: %.4f ± %.4f\n', avg_mu_with_bias(end), se_mu_with_bias(end));

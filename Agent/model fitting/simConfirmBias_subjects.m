% MATLAB script for running 100 simulations with and without confirmation bias and plotting the average mu

clc
clearvars

data = importdata("preprocessed_dataFitting.mat");
% Set number of simulations
numSubjs = length(unique(data.ID));
subjIDs = unique(data.ID);

% Initialize storage arrays (assuming mu has a certain length - adjust as needed)
% You may need to adjust the second dimension based on your actual mu length
mu_no_bias = [];
mu_with_bias = [];

% Run simulations without confirmation bias
for i = 1:numSubjs
    dataSubj = data(data.ID == subjIDs(i),:);
    blockIDs = unique(dataSubj.blocks);
    numBlocks = length(blockIDs);
    for b = 5:numBlocks
        agent = Agent();
        dataBlocks = dataSubj(dataSubj.blocks == blockIDs(b),:);

        for t = 1:height(dataBlocks)
            agent.confirmation_bias = dataBlocks.confirm_rew(t);
            agent.o_t = dataBlocks.con_diff_choice(t);
            agent.a_t = dataBlocks.choice(t);
            agent.r_t = dataBlocks.recoded_reward(t);
            agent.learn(dataBlocks.recoded_reward(t));
            disp(agent.sigma)
            dataBlocks.mu_agent(t) = agent.G;
        end

    end
    % sim_result = task_agent_int(1,1,0);
    % if i == 1
    %     % Initialize arrays based on actual mu length
    %     mu_no_bias = zeros(numSubjs, length(sim_result.mu));
    %     mu_with_bias = zeros(numSubjs, length(sim_result.mu));
    % end
    % mu_no_bias(i, :) = sim_result.mu;
    % fprintf('No bias simulation %d/%d completed\n', i, numSubjs);
end

% Run simulations with confirmation bias
agent = Agent();
agent.confirmation_bias = 1;
for i = 1:numSubjs
    sim_result = task_agent_int(1,1,0);
    mu_with_bias(i, :) = sim_result.mu;
    fprintf('With bias simulation %d/%d completed\n', i, numSubjs);
end

% Calculate average mu across simulations
avg_mu_no_bias = mean(mu_no_bias, 1);
avg_mu_with_bias = mean(mu_with_bias, 1);

% Calculate standard error for error bars (optional)
se_mu_no_bias = std(mu_no_bias, 0, 1) / sqrt(numSubjs);
se_mu_with_bias = std(mu_with_bias, 0, 1) / sqrt(numSubjs);

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
title(sprintf('Average mu across %d simulations', numSubjs));
legend('Location', 'best');
grid on;
hold off;

% Display summary statistics
fprintf('\nSummary Statistics:\n');
fprintf('No Confirmation Bias - Final mu: %.4f ± %.4f\n', avg_mu_no_bias(end), se_mu_no_bias(end));
fprintf('With Confirmation Bias - Final mu: %.4f ± %.4f\n', avg_mu_with_bias(end), se_mu_with_bias(end));

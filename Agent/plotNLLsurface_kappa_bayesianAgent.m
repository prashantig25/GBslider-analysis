clc
clearvars

data = importdata("preprocessed_dataFitting.mat");
blocks = data.blocks;
state = data.state;
subjIDs = unique(data.ID);
numSubjs = length(subjIDs);
%
% mu_hat = rand(height(data),1);

% INITIALIZE
kappa = randi([1, 100], 1, 50);
n_blocks = 12;
trials_per_block = 25;
alpha = 0.1;

% Set up a range of kappa values to test
kappa_grid = linspace(1, 100, 100); % or use your actual bounds

% Fix alpha and beta at reasonable values (e.g., from your initial guess or previous fits)
alpha_fixed = 0.1;

nll_vals = NaN(length(kappa_grid),numSubjs);

figure("Position",[200,200,800,500])
hold on 
for n = 1:25%numSubjs

    % GET THE DATA REQUIRED FOR THIS SUBJECT
    dataSubj = data(data.ID == subjIDs(n),:);
    % dataSubj(dataSubj.choice_cond == 2,:) = [];
    dataSubj(dataSubj.choice_cond == 3,:) = [];
    % dataSubj(dataSubj.contrast == 1,:) = [];
    dataSubj.condiff_relative = (dataSubj.contrast_left - dataSubj.contrast_right)./2;
    for h = 1:height(dataSubj)
        if dataSubj.contrast(h) == 0
            rewards(h) = dataSubj.correct(h);%(dataSubj.blocks == 1); % Vector of outcomes
        else
            rewards(h) = 1-dataSubj.correct(h);%(dataSubj.blocks == 1); % Vector of outcomes
        end
    end
    
    % GET ALL RELEVANT INFO
    blocks = dataSubj.blocks;
    state = dataSubj.state;
    choices = dataSubj.choice;
    condiff = dataSubj.condiff_relative;

    % SIMULATE THE TRUE MU USING FIXED KAPPA
    [trueMU] = simulate_kappa_BayesianAgent(kappa(n), blocks, length(unique(dataSubj.blocks)), 25, dataSubj, rewards.');
    mu_hatSubj = trueMU;

    % GET THE NLL VALS FOR THIS SIMULATED MU FOR THE ENTIRE KAPPA RANGE
    for i = 1:length(kappa_grid)
        params = [kappa_grid(i), 0.04, 25];
        nll_vals(i,n) = nll_fitSlider_bayesianAgent(params, mu_hatSubj, dataSubj, ...
            8, 25, unique(dataSubj.blocks), rewards.'); % mu_hat is your data
    end

    % PLOT
    subplot(5,5,n)
    hold on
    plot(kappa_grid, nll_vals(:,n), 'LineWidth', 1, 'Color', 'b');
    hold on
    scatter(kappa_grid(kappa_grid == kappa(n)),nll_vals(kappa(n),n),"filled", ...
        "o","MarkerEdgeColor",'k','MarkerFaceColor',[1,0.7,0.4],'MarkerFaceAlpha',0.7,'SizeData',80)
    xlabel('\kappa');
    ylabel('Negative Log-Likelihood');
    title('LL Surface for \kappa');
    grid on;
end

hold on
sgtitle('Both and perceptual condition')



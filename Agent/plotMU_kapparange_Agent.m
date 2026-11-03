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
kappa = linspace(1,100,20);
n_blocks = 12;
trials_per_block = 25;
alpha = 0.1;

% Set up a range of kappa values to test
kappa_grid = linspace(1, 100, 20); % or use your actual bounds
cmap = winter(length(kappa));  % Blue-ish colormap
nSimulations = 20;
estError_kappa = NaN(length(kappa_grid),nSimulations);

figure("Position",[200,200,800,500])
hold on 

% GET THE DATA REQUIRED FOR THIS SUBJECT
dataSubj = data(data.ID == subjIDs(1),:);

for s = 1:nSimulations
for n = 1:length(kappa)%numSubjs


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

    % PLOT
    hold on
    plot(1:25,trueMU(1:25),'Color', cmap(n,:), 'LineWidth', 0.5)

    % CALCULATE THE ESTIMATION ERROR
    estError_kappa(n,s) = nanmean(abs(0.7-trueMU(1:25)));

end
end

hold on
legend(string(kappa_grid))
sgtitle('Both and perceptual condition')

% Average estimation error across simulations
meanEstError = mean(estError_kappa, 2, 'omitnan');

% Set up figure
figure("Position",[200,200,900,500])
hold on

% Bar plot of mean estimation error
bh = bar(1:length(kappa_grid), meanEstError, 'FaceColor', [0.9 0.9 0.7], ...
    'EdgeColor', 'k', 'LineWidth', 1.2);
ylabel('Mean Absolute Estimation Error (|0.9 - \mu|)')
xlabel('\kappa')
xticks(1:length(kappa_grid))
xticklabels(round(kappa_grid, 1))
xtickangle(45)
title('Estimation Error vs \kappa')
grid on

% Overlay individual datapoints (jittered for visibility)
for i = 1:length(kappa_grid)
    x = repmat(i, nSimulations, 1) + 0.15*randn(nSimulations,1);  % jitter
    y = estError_kappa(i,:)';
    scatter(x, y, 40, 'filled', 'MarkerFaceColor', [0.1 0.1 0.9], ...
        'MarkerFaceAlpha', 0.5)
end

legend({'Mean Estimation Error', 'Simulation Samples'}, 'Location', 'northeast')



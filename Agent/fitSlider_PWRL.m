
clc
clearvars

data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);

alphaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);

dataContrast = data;
% dataContrast = data(data.contrast ==0,:);
dataContrast = dataContrast(dataContrast.choice_cond ~=3,:);
% dataContrast = dataContrast(dataContrast.choice_cond ~=1,:);
dataContrast.condiff_relative = (dataContrast.contrast_left - dataContrast.contrast_right)./2;

% Initial parameter guesses
% init_params = [0.1, 0.5, 5]; % [alpha, beta]
init_params = [0.1, 15, 0.01]; % [alpha, beta]

% Parameter bounds
% lb = [0  0   1];
% ub = [1  20 100];
lb = [0  1 0];
ub = [1  100 0.1];

for n = 1:numSubjs

    dataSubj = dataContrast(dataContrast.ID == uniqueID(n),:);

    % Prepare your data
    mu_hat = dataSubj.mu_congruence; % slider reported probabilities (N_trials x 1)
    dataSubj(isnan(mu_hat),:) = []; % remove all trials with mu nans
    mu_hat = mu_hat(~isnan(mu_hat)); % remove nans from data
    blocks = dataSubj.blocks;
    state = dataSubj.state;
    condiff = dataSubj.condiff_relative;
    % choices = dataSubj.choice; %(dataSubj.blocks == 1); % Vector of subject's choices

    % for h = 1:height(dataSubj)
    %     if dataSubj.contrast(h) == 0
            recoded_rewards = dataSubj.recoded_reward;%(dataSubj.blocks == 1); % Vector of outcomes
    %     else
    %         recoded_rewards(h) = dataSubj.recoded_reward(h);%(dataSubj.blocks == 1); % Vector of outcomes
    %     end
    % end

    rewards = dataSubj.correct;
    choices = dataSubj.choice;
    contrast = dataSubj.contrast;

    % Negative log-likelihood function
    nll_fun = @(params) nll_fitSlider_PWRL(params, mu_hat, blocks, rewards, condiff, choices, recoded_rewards, contrast);
    
    % Minimize negative log-likelihood
    options = optimset('Display', 'iter');
    [params, nll_min] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    
    alphaParameter(n,1) = params(1);
    sigmaParameter(n,1) = params(3);
    kappaParameter(n,1) = params(2);
    ecoperf(n,1) = nanmean(dataSubj.ecoperf);
    esterror(n,1) = nanmean(dataSubj.est_error);
end


y = [kappaParameter];
mean_all = [mean(kappaParameter)];
SEM_all = [nanstd(kappaParameter)./sqrt(numSubjs)];
n = numSubjs;
x_groups = 1;
bars = 1;
legend_names = {'',''};
xticks = [1];
xticklabs = ["kappa parameter"];
title_name = 'PWRL (Larsen) slider model fit (Both and perceptual conditions)';
xlabelname = ''; ylabelname = ''; fontsize = 12; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])
% ylim([0,50])


y = [alphaParameter];
mean_all = [mean(alphaParameter)];
SEM_all = [nanstd(alphaParameter)./sqrt(numSubjs)];
n = numSubjs;
x_groups = 1;
bars = 1;
legend_names = {'',''};
xticks = [1];
xticklabs = ["alpha parameter"];
title_name = 'PWRL (Larsen) slider model fit (Both and perceptual conditions)';
xlabelname = ''; ylabelname = ''; fontsize = 12; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])
% ylim([0,100])

y = [sigmaParameter];
mean_all = [mean(sigmaParameter)];
SEM_all = [nanstd(sigmaParameter)./sqrt(numSubjs)];
n = numSubjs;
x_groups = 1;
bars = 1;
legend_names = {'',''};
xticks = [1];
xticklabs = ["sigma parameter"];
title_name = 'PWRL (Larsen) slider model fit (Both and perceptual conditions)';
xlabelname = ''; ylabelname = ''; fontsize = 12; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])

 figure
hold on
scatter(esterror,sigmaParameter)
lsline
title(num2str(corr(esterror,sigmaParameter,"type","Spearman")))
xlabel('Economic performance (All Conditions)')
ylabel('sigma parameter')

figure
hold on
scatter(esterror,alphaParameter)
lsline
title(num2str(corr(esterror,alphaParameter,"type","Spearman")))
xlabel('Estimation error (All Conditions)')
ylabel('alpha parameter')

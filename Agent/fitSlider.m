% PG notes: estimate the kappa concentration parameter
% start with the mu reported by participant on first trial and an initial value for kappa
% then get a = mu*kappa and b = (1-mu)*kappa
% then evaluate x = 0:0.01:1 with a and b
% get the mean of that distribution which is the q-values ??
% then get estimate beta using softmax
% then decrease nll
% then fit the alpha parameter using q-values ?
% loop over trials

clc
clearvars

data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);

alphaParameter = NaN(numSubjs, 1);
betaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);

dataContrast = data;
% dataContrast = data(data.contrast ==0,:);
% dataContrast = dataContrast(dataContrast.choice_cond == 1,:);
dataContrast.condiff_relative = (dataContrast.contrast_left - dataContrast.contrast_right)./2;

% Initial parameter guesses
% init_params = [0.1, 0.5, 5]; % [alpha, beta]
init_params = [0.1, 5]; % [alpha, beta]

% Parameter bounds
% lb = [0  0   1];
% ub = [1  20 100];
lb = [0  1];
ub = [1  100];

for n = 1:numSubjs
    dataSubj = dataContrast(dataContrast.ID == uniqueID(n),:);

    % Prepare your data
    mu_hat = dataSubj.mu_congruence; % slider reported probabilities (N_trials x 1)
    mu_hat = mu_hat(~isnan(mu_hat)); % remove nans from data
    blocks = dataSubj.blocks;
    state = dataSubj.state;
    % choices = dataSubj.choice; %(dataSubj.blocks == 1); % Vector of subject's choices

    for h = 1:height(dataSubj)
        if dataSubj.contrast(h) == 0
            rewards(h) = dataSubj.recoded_reward(h);%(dataSubj.blocks == 1); % Vector of outcomes
        else
            rewards(h) = 1-dataSubj.recoded_reward(h);%(dataSubj.blocks == 1); % Vector of outcomes
        end
    end

    % Set parameter constraints (e.g., learning rate between 0 and 1)
    % params = fit_RL_model(choices, rewards, state, condiff, contrast);

    % Negative log-likelihood function
    nll_fun = @(params) nll_fitSlider_basicRL(params, mu_hat, blocks, state, rewards);
    
    % Minimize negative log-likelihood
    options = optimset('Display', 'iter');
    [params, nll_min] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    
    alphaParameter(n,1) = params(1);
    % betaParameter(n,1) = params(2);
    kappaParameter(n,1) = params(2);
    ecoperf(n,1) = nanmean(dataSubj.ecoperf);
    esterror(n,1) = nanmean(dataSubj.est_error);
end


% y = [alphaParameter; betaParameter; kappaParameter];
% mean_all = [mean(alphaParameter), mean(betaParameter), mean(kappaParameter)];
% SEM_all = [nanstd(alphaParameter)./sqrt(numSubjs), nanstd(betaParameter)./sqrt(numSubjs), nanstd(kappaParameter)./sqrt(numSubjs)];
% n = numSubjs;
% x_groups = 3;
% bars = 1;
% legend_names = {'',''};
% xticks = [1:3];
% xticklabs = ["LR", "beta parameter", "kappa parameter"];
% title_name = 'Simple RL model slider fit (All Conditions)';
% xlabelname = ''; ylabelname = ''; fontsize = 8; linewidth = 1; fontname = 'Arial';
% 
% figure
% hold on
% bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
%     xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])
% ylim([0,100])

y = [kappaParameter];
mean_all = [mean(kappaParameter)];
SEM_all = [nanstd(kappaParameter)./sqrt(numSubjs)];
n = numSubjs;
x_groups = 1;
bars = 1;
legend_names = {'',''};
xticks = [1];
xticklabs = ["kappa parameter"];
title_name = 'Simple RL slider model fit (All conditions)';
xlabelname = ''; ylabelname = ''; fontsize = 8; linewidth = 1; fontname = 'Arial';

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
title_name = 'Simple RL slider model fit (All conditions)';
xlabelname = ''; ylabelname = ''; fontsize = 8; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])
% ylim([0,100])

figure
hold on
scatter(ecoperf,kappaParameter)
lsline
title(num2str(corr(ecoperf,kappaParameter,"type","Spearman")))
xlabel('Economic performance (All Conditions)')
ylabel('kappa parameter')

figure
hold on
scatter(esterror,alphaParameter)
lsline
title(num2str(corr(esterror,alphaParameter,"type","Spearman")))
xlabel('Estimation error (All Conditions)')
ylabel('alpha parameter')

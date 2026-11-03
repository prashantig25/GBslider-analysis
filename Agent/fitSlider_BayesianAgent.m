
clc
clearvars

data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);

betaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);

dataContrast = data;
% dataContrast = data(data.contrast ==0,:);
% dataContrast = dataContrast(dataContrast.choice_cond ~=1,:);
dataContrast = dataContrast(dataContrast.choice_cond ~=3,:);
dataContrast.condiff_relative = (dataContrast.contrast_left - dataContrast.contrast_right)./2;

% Initial parameter guesses
init_params = [5, 0.05]; % [kappa, sigma, beta]
% init_params = [5, 0.01]; % [alpha, beta]
% init_params = [5]; % [alpha, beta]

% Parameter bounds
lb = [1  0];
ub = [100  0.1];

for n = 1:numSubjs
    dataSubj = dataContrast(dataContrast.ID == uniqueID(n),:);

    % Prepare your data
    mu_hat = dataSubj.mu_congruence; % slider reported probabilities (N_trials x 1)
    mu_hat = mu_hat(~isnan(mu_hat)); % remove nans from data
    blocks = dataSubj.blocks(~isnan(mu_hat));
    state = dataSubj.state(~isnan(mu_hat));
    condiff = dataSubj.condiff_relative(~isnan(mu_hat));
    choices = dataSubj.choice(~isnan(mu_hat)); %(dataSubj.blocks == 1); % Vector of subject's choices
    correct = dataSubj.correct(~isnan(mu_hat));
    contrast = dataSubj.contrast(~isnan(mu_hat));
    for h = 1:height(correct)
        if contrast(h) == 0
            rewards(h) = correct(h);%(dataSubj.blocks == 1); % Vector of outcomes
        else
            rewards(h) = 1-correct(h);%(dataSubj.blocks == 1); % Vector of outcomes
        end
    end
    dataSubj(isnan(dataSubj.mu_congruence),:) = [];

    % Negative log-likelihood function
    nll_fun = @(params) nll_fitSlider_bayesianAgent(params, mu_hat, dataSubj, length(unique(dataSubj.blocks)), 25, unique(dataSubj.blocks), rewards.');
    
    % Minimize negative log-likelihood
    options = optimset('Display', 'iter');
    [params, nll_min] = fmincon(nll_fun, init_params, [], [], [], [], lb, ub, [], options);
    
    sigmaParameter(n,1) = params(2);
    kappaParameter(n,1) = params(1);
    % betaParameter(n,1) = params(3);
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
title_name = 'Agent + slider model fit (All conditions)';
xlabelname = ''; ylabelname = ''; fontsize = 12; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])
% ylim([0,50])


y = [sigmaParameter];
mean_all = [mean(sigmaParameter)];
SEM_all = [nanstd(sigmaParameter)./sqrt(numSubjs)];
n = numSubjs;
x_groups = 1;
bars = 1;
legend_names = {'',''};
xticks = [1];
xticklabs = ["sigma parameter"];
title_name = 'Agent + slider model fit (All conditions)';
xlabelname = ''; ylabelname = ''; fontsize = 12; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])

y = [betaParameter];
mean_all = [mean(betaParameter)];
SEM_all = [nanstd(betaParameter)./sqrt(numSubjs)];
n = numSubjs;
x_groups = 1;
bars = 1;
legend_names = {'',''};
xticks = [1];
xticklabs = ["beta parameter"];
title_name = 'Agent + slider model fit (Perceptual conditions)';
xlabelname = ''; ylabelname = ''; fontsize = 12; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])

figure
hold on
scatter(esterror,kappaParameter)
lsline
title(num2str(corr(esterror,kappaParameter,"type","Spearman")))
xlabel('Economic performance (All Conditions)')
ylabel('kappa parameter')

figure
hold on
scatter(esterror,sigmaParameter)
lsline
title(num2str(corr(esterror,sigmaParameter,"type","Spearman")))
xlabel('Estimation error (All Conditions)')
ylabel('sigma parameter')

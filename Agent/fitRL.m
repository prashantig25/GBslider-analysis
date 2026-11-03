clc
clearvars

data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);

alphaParameter = NaN(numSubjs, 1);
betaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
ecoperf = NaN(numSubjs,1);

dataContrast = data;
% dataContrast = data(data.contrast ==1,:);
dataContrast = dataContrast(dataContrast.choice_cond == 1,:);
dataContrast.condiff_relative = (dataContrast.contrast_left - dataContrast.contrast_right)./2;

for n = 1:numSubjs
    dataSubj = dataContrast(dataContrast.ID == uniqueID(n),:);

    % Prepare your data
    choices = dataSubj.choice;%(dataSubj.blocks == 1); % Vector of subject's choices

    for h = 1:height(dataSubj)
        if dataSubj.contrast(h) == 0
            rewards(h) = dataSubj.recoded_reward(h);%(dataSubj.blocks == 1); % Vector of outcomes
        else
            rewards(h) = 1-dataSubj.recoded_reward(h);%(dataSubj.blocks == 1); % Vector of outcomes
        end
    end
    state = dataSubj.state;%(dataSubj.blocks == 1);
    condiff = dataSubj.condiff_relative;
    contrast = dataSubj.contrast;

    % Set parameter constraints (e.g., learning rate between 0 and 1)
    params = fit_RL_model(choices, rewards, state, condiff, contrast);

    alphaParameter(n,1) = params(1);
    betaParameter(n,1) = params(2);
    sigmaParameter(n,1) = params(3);
    ecoperf(n,1) = nanmean(dataSubj.ecoperf);
end

y = [alphaParameter; betaParameter; sigmaParameter];
mean_all = [mean(alphaParameter), mean(betaParameter), mean(sigmaParameter)];
SEM_all = [nanstd(alphaParameter)./sqrt(numSubjs), nanstd(betaParameter)./sqrt(numSubjs), nanstd(sigmaParameter)./sqrt(numSubjs)];
n = numSubjs;
x_groups = 3;
bars = 1;
legend_names = {'',''};
xticks = [1:3];
xticklabs = ["LR", "beta parameter", "sigma parameter"];
title_name = 'Simple RL model fit (All Conditions)';
xlabelname = ''; ylabelname = ''; fontsize = 8; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])
ylim([0,10])

y = [sigmaParameter];
mean_all = [mean(sigmaParameter)];
SEM_all = [nanstd(sigmaParameter)./sqrt(numSubjs)];
n = numSubjs;
x_groups = 1;
bars = 1;
legend_names = {'',''};
xticks = [1];
xticklabs = ["sigma parameter"];
title_name = 'Simple RL model fit (Both condition)';
xlabelname = ''; ylabelname = ''; fontsize = 8; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])
ylim([0,0.1])

figure
hold on
scatter(ecoperf,betaParameter)
lsline
title(num2str(corr(ecoperf,betaParameter,"type","Spearman")))
xlabel('Economic performance (All Conditions)')
ylabel('Beta parameter')

figure
hold on
scatter(ecoperf,sigmaParameter)
lsline
title(num2str(corr(ecoperf,sigmaParameter,"type","Spearman")))
xlabel('Economic performance (All Conditions)')
ylabel('Sigma parameter')

safe_saveall("alphaBoth_simpleRL.mat",alphaParameter)
safe_saveall("betaBoth_simpleRL.mat",betaParameter)
safe_saveall("sigmaBoth_simpleRL.mat",sigmaParameter)
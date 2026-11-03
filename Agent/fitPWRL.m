clc
clearvars

data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);

alphaParameter = NaN(numSubjs, 1); % alpha in ezizizi 2023
TParameter = NaN(numSubjs, 1); % T parameter in ezizizi 2023
rhoParameter = NaN(numSubjs, 1); %NaN(numSubjs, 1); % rho parameter in ezizizi 2023
ecoperf = NaN(numSubjs,1);

dataContrast = data(data.contrast ==0,:);
dataContrast = dataContrast(dataContrast.choice_cond ~= 2,:);

dataContrast.recoded_rewardState = dataContrast.recoded_reward;
for h = 1:height(dataContrast)
    if dataContrast.state(h) == 1
        dataContrast.recoded_rewardState(h) = 1-dataContrast.recoded_reward(h);
    end
end

for n = 1:numSubjs
    dataSubj = dataContrast(dataContrast.ID == uniqueID(n),:);

    % Prepare your data
    choices = dataSubj.choice;%(dataSubj.blocks == 1); % Vector of subject's choices
    rewards = dataSubj.correct;%(dataSubj.blocks == 1); % Vector of outcomes
    state = dataSubj.state;%(dataSubj.blocks == 1);
%     condiff = dataSubj.con_diff_choice;

    % Set parameter constraints (e.g., learning rate between 0 and 1)
    params = fit_PWRL_model(choices, rewards, state);

    alphaParameter(n,1) = params(1);
    TParameter(n,1) = params(2);
    rhoParameter(n,1) = params(3);
    ecoperf(n,1) = nanmean(dataSubj.ecoperf);
end

% safe_saveall("alphaParameter_AdaptiveLR_onlyReward.mat",alphaParameter);
% safe_saveall("betaParameter_AdaptiveLR_onlyReward.mat",TParameter);
% safe_saveall("betaParameter_AdaptiveLR_onlyReward.mat",rhoParameter);

y = [alphaParameter];
mean_all = [mean(alphaParameter)];
SEM_all = [nanstd(alphaParameter)./sqrt(numSubjs)];
n = numSubjs;
x_groups = 1;
bars = 1;
legend_names = {'',''};
xticks = [1];
xticklabs = ["LR", "beta parameter"];
title_name = 'PWRL model fit (Only perceptual condition)';
xlabelname = ''; ylabelname = ''; fontsize = 8; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])
ylim([0,1])

y = [TParameter];
mean_all = [mean(TParameter)];
SEM_all = [nanstd(TParameter)./sqrt(numSubjs)];
n = numSubjs;
x_groups = 1;
bars = 1;
legend_names = {'',''};
xticks = [1];
xticklabs = ["T (temperature) parameter"];
title_name = 'PWRL model fit (Only perceptual condition)';
xlabelname = ''; ylabelname = ''; fontsize = 8; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])
ylim([0,1])

y = [rhoParameter];
mean_all = [mean(rhoParameter)];
SEM_all = [nanstd(rhoParameter)./sqrt(numSubjs)];
n = numSubjs;
x_groups = 1;
bars = 1;
legend_names = {'',''};
xticks = [1];
xticklabs = ["sensitivity (rho) parameter"];
title_name = 'PWRL model fit (No perceptual condition)';
xlabelname = ''; ylabelname = ''; fontsize = 8; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])
ylim([0,1])


figure
hold on
scatter(ecoperf,TParameter)
lsline
title(num2str(corr(ecoperf,TParameter,"type","Spearman")))
xlabel('Economic performance (Only perceptual condition)')
ylabel('T parameter')

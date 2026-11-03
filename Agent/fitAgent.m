% FIT AGENT TO CHOICE DATA
% what parameters do we want to estimate?
% beta based on the choices and the decreasing loglikelihood

clc
clearvars

% INITIALISE STUFF
dataAll = importdata("preprocessed_dataFitting.mat");
dataAll.condiff_relative = (dataAll.contrast_left - dataAll.contrast_right)./2;
dataAll = dataAll(dataAll.choice_cond ~= 3,:);
% dataAll = dataAll(dataAll.contrast == 0,:);
numSubj = length(unique(dataAll.ID));
uniqueIDs = unique(dataAll.ID);
sigmaAll = importdata("sigmaNoReward_simpleRL.mat"); % take in the estimated sigma parameter
betaParameter = NaN(numSubj,1);
sigmaParameter = NaN(numSubj,1);

for n = 1:numSubj

    sigma = sigmaAll(n);
    data = dataAll(dataAll.ID == uniqueIDs(n),:);
    params = fitBayesianAgent(data, sigma);
    betaParameter(n,1) = params(1);
    sigmaParameter(n,1) = params(2);
    ecoperf(n,1) = nanmean(data.ecoperf);
    fprintf('Participant number: %d\n...', n);

end

y = [betaParameter];
mean_all = [mean(betaParameter)];
SEM_all = [nanstd(betaParameter)./sqrt(numSubj)];
n = numSubj;
x_groups = 1;
bars = 1;
legend_names = {'',''};
xticks = [1];
xticklabs = ["Beta parameter"];
title_name = 'Fit agent (No reward)';
xlabelname = ''; ylabelname = ''; fontsize = 8; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])
ylim([0,10])

y = [sigmaParameter];
mean_all = [mean(sigmaParameter)];
SEM_all = [nanstd(sigmaParameter)./sqrt(numSubj)];
n = numSubj;
x_groups = 1;
bars = 1;
legend_names = {'',''};
xticks = [1];
xticklabs = ["Sigma parameter"];
title_name = 'Fit agent (No reward)';
xlabelname = ''; ylabelname = ''; fontsize = 8; linewidth = 1; fontname = 'Arial';

figure
hold on
bar_plots(y,mean_all.',SEM_all.',n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,fontsize,linewidth,fontname,[0,0,0.7])
ylim([0,10])

figure
hold on
scatter(ecoperf,betaParameter)
lsline
title(num2str(corr(ecoperf,betaParameter,"type","Spearman")))
xlabel('Economic performance (No reward)')
ylabel('Beta parameter')

safe_saveall("estBeta_NoReward_bothContrasts_withSigma.mat",betaParameter)
safe_saveall("estSigma_NoReward_bothContrasts.mat",sigmaParameter)
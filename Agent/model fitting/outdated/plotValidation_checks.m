clc
clearvars

simData = importdata("fullSims_allmodels.mat");
data = importdata("preprocessed_dataFitting.mat");
data = data(data.choice_cond ~= 3,:);
data(data.pe == 0,:) = [];
data.lr = data.up./data.pe;
data(abs(data.lr) > 2, :) = [];
unique_ids = unique(data.ID);
numSubjs = length(unique_ids);
numSims = 200;

simPE_all = cell(numSubjs,1);
simUP_all = cell(numSubjs,1);

for n = 1:numSubjs

    simData_subj = simData(n);
    numTrials = length(simData_subj.Trials);
    simPe_subj = NaN(numTrials,1);
    simUp_subj = NaN(numTrials,numSims);
    simUpmean_subj = NaN(numTrials,1);
    for t = 1:numTrials 
        trialData = simData_subj.Trials(t);

        % get pe
        simPe_subj(t,1) = trialData.('basicRL').SimPE;

        % get updates
        simUp_subj(t,:) = trialData.('BayesianAgent').SimUpdates;

        % get mean updates
        simUpmean_subj(t,1) = nanmean(simUp_subj(t,:));

    end

    simPE_all{n,1} = simPe_subj;
    simUP_all{n,1} = simUpmean_subj;

end

% get PE bins
PEbins = linspace(0,0.1,10);
% PEbins = linspace(-1,1,21);
upBinned = NaN(length(PEbins),numSubjs);

% plot means for each bin across participants
for n = 1:numSubjs
    dataSubj = data(data.ID == unique_ids(n),:);
    subjPE = dataSubj.pe;
    condiff = dataSubj.con_diff_choice;
    % bins = discretize(simPE_all{n,1},PEbins);
    % bins = discretize(subjPE,PEbins);
    bins = discretize(condiff,PEbins);
    simUp = abs(simUP_all{n,1});
    for b = 1:length(PEbins)
        upBinned(b,n) = nanmean(simUp(bins == b));
    end

end

figure
hold on
errorbar(PEbins, nanmean(upBinned,2).',nanstd(upBinned,0,2)./ ...
    sqrt(numSubjs),'Marker','o')